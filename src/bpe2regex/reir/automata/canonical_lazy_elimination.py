"""Budgeted lazy lowering from persistent adjacency to a boundary regex."""

import time
from dataclasses import dataclass
from typing import Any

from ...tagged_fst import TaggedFST
from ..builder import DEFAULT_BUILDER
from ..cost import raw_deflate_size
from ..marked import BOUNDARY, verify_single_boundary
from ..marked_source import render_marked_regex
from ..ops import EPSILON, Op
from ..tagged_source import render_tagged_regex
from .canonical_adjacency import (
    CanonicalAdjacencyCompiler,
    CanonicalAdjacencyMetrics,
)
from .canonical_lazy import (
    CanonicalLazyQuotient,
    CanonicalLazyQuotientCompiler,
    CanonicalLazyQuotientMetrics,
)
from .canonical_regex import TokenSymbolLowerer
from .elimination import GeneralizedAutomaton
from .labels import SymbolSet


@dataclass(frozen=True, slots=True)
class LazyEliminationBudget:
    max_states: int | None = 512
    max_symbol_checks: int | None = 5_000_000
    max_transition_groups: int | None = 1_000_000
    max_intermediate_edges: int | None = 1_000_000
    max_expression_occurrences: int | None = 5_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_states",
            "max_symbol_checks",
            "max_transition_groups",
            "max_intermediate_edges",
            "max_expression_occurrences",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class LazyEliminationMetrics:
    quotient_state_count: int
    active_token_count: int
    materialized_row_count: int
    symbol_checks: int
    transition_group_count: int
    initial_edge_count: int
    peak_edge_count: int
    eliminated_state_count: int
    elimination_order: tuple[int, ...]
    elapsed_seconds: float


class LazyEliminationBudgetExceeded(RuntimeError):
    def __init__(self, reason: str, metrics: LazyEliminationMetrics) -> None:
        super().__init__(f"lazy elimination budget exceeded: {reason}")
        self.reason = reason
        self.metrics = metrics


@dataclass(frozen=True, slots=True)
class LazyBoundaryEliminationResult:
    expression: Op
    metrics: LazyEliminationMetrics


def _strongly_connected_components(
    successors: tuple[tuple[int, ...], ...],
) -> tuple[frozenset[int], ...]:
    state_count = len(successors)
    predecessors: list[list[int]] = [[] for _ in range(state_count)]
    for source, row in enumerate(successors):
        for target in row:
            predecessors[target].append(source)
    for row in predecessors:
        row.sort()

    visited: set[int] = set()
    finish_order: list[int] = []
    for root in range(state_count):
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            state, index = stack[-1]
            if index < len(successors[state]):
                target = successors[state][index]
                stack[-1] = state, index + 1
                if target not in visited:
                    visited.add(target)
                    stack.append((target, 0))
                continue
            finish_order.append(state)
            stack.pop()

    assigned: set[int] = set()
    components: list[frozenset[int]] = []
    for root in reversed(finish_order):
        if root in assigned:
            continue
        component: set[int] = set()
        pending = [root]
        assigned.add(root)
        while pending:
            state = pending.pop()
            component.add(state)
            for source in reversed(predecessors[state]):
                if source not in assigned:
                    assigned.add(source)
                    pending.append(source)
        components.append(frozenset(component))
    components.reverse()
    return tuple(components)


def _bounded_occurrences(root: Op, limit: int | None) -> int:
    cache: dict[int, tuple[Op, int]] = {}

    def visit(op: Op) -> int:
        known = cache.get(id(op))
        if known is not None and known[0] is op:
            return known[1]
        count = 1
        for child in op.operands:
            count += visit(child)
            if limit is not None and count > limit:
                break
        cache[id(op)] = (op, count)
        return count

    return visit(root)


class CanonicalLazyBoundaryEliminator:
    """Stream quotient rows into a GNFA and stop before an unsafe expansion."""

    def __init__(self, budget: LazyEliminationBudget | None = None) -> None:
        self.budget = LazyEliminationBudget() if budget is None else budget

    def lower(
        self,
        quotient: CanonicalLazyQuotient,
        tokens: tuple[bytes | None, ...],
    ) -> LazyBoundaryEliminationResult:
        if quotient.alphabet_size != len(tokens):
            raise ValueError("lazy quotient and vocabulary have different alphabets")
        started = time.perf_counter()
        rows = 0
        checks = 0
        groups = 0
        initial_edges = 0
        peak_edges = 0
        eliminated = 0
        order: tuple[int, ...] = ()

        def metrics() -> LazyEliminationMetrics:
            return LazyEliminationMetrics(
                quotient.state_count,
                len(quotient.active_tokens),
                rows,
                checks,
                groups,
                initial_edges,
                peak_edges,
                eliminated,
                order,
                time.perf_counter() - started,
            )

        def require(condition: bool, reason: str) -> None:
            if not condition:
                raise LazyEliminationBudgetExceeded(reason, metrics())

        if self.budget.max_states is not None:
            require(
                quotient.state_count <= self.budget.max_states,
                "quotient state count",
            )
        projected_checks = quotient.state_count * len(quotient.active_tokens)
        if self.budget.max_symbol_checks is not None:
            require(
                projected_checks <= self.budget.max_symbol_checks,
                "symbol checks",
            )

        materialized_rows = []
        successors: list[tuple[int, ...]] = []
        for state in range(quotient.state_count):
            row = quotient.transition_groups(state)
            materialized_rows.append(row)
            rows += 1
            checks += len(quotient.active_tokens)
            groups += len(row)
            if self.budget.max_transition_groups is not None:
                require(
                    groups <= self.budget.max_transition_groups,
                    "transition group count",
                )
            successors.append(tuple(sorted(transition.target for transition in row)))

        source = quotient.state_count
        final = source + 1
        graph = GeneralizedAutomaton(
            (*range(quotient.state_count), source, final),
            source,
            final,
        )
        symbol_lowerer = TokenSymbolLowerer(tokens)

        def lower_tokens(ranks: tuple[int, ...]) -> Op:
            return symbol_lowerer(SymbolSet.from_symbols(quotient.alphabet_size, ranks))

        for transition in materialized_rows[quotient.start]:
            graph.add_edge(
                source,
                transition.target,
                DEFAULT_BUILDER.concat(lower_tokens(transition.tokens), BOUNDARY),
            )
        for state, row in enumerate(materialized_rows):
            for transition in row:
                graph.add_edge(
                    state,
                    transition.target,
                    lower_tokens(transition.tokens),
                )
            graph.add_edge(state, final, EPSILON)
        initial_edges = len(graph.edges)
        peak_edges = initial_edges
        if self.budget.max_intermediate_edges is not None:
            require(
                initial_edges <= self.budget.max_intermediate_edges,
                "initial GNFA edge count",
            )

        order = tuple(
            state
            for component in _strongly_connected_components(tuple(successors))
            for state in sorted(component)
        )
        for state in order:
            graph.eliminate(state)
            eliminated += 1
            peak_edges = max(peak_edges, len(graph.edges))
            if self.budget.max_intermediate_edges is not None:
                require(
                    len(graph.edges) <= self.budget.max_intermediate_edges,
                    "intermediate GNFA edge count",
                )
            if self.budget.max_expression_occurrences is not None:
                occurrences = sum(
                    _bounded_occurrences(
                        expression,
                        self.budget.max_expression_occurrences,
                    )
                    for expression in graph.edges.values()
                )
                require(
                    occurrences <= self.budget.max_expression_occurrences,
                    "expression occurrences",
                )
        expression = graph.expression
        verify_single_boundary(expression)
        return LazyBoundaryEliminationResult(expression, metrics())


@dataclass(frozen=True, slots=True)
class CanonicalLazyBoundaryMetrics:
    adjacency: CanonicalAdjacencyMetrics
    quotient: CanonicalLazyQuotientMetrics
    elimination: LazyEliminationMetrics


@dataclass(frozen=True, slots=True)
class CanonicalLazyBoundaryIRResult:
    expression: Op
    quotient: CanonicalLazyQuotient
    metrics: CanonicalLazyBoundaryMetrics


@dataclass(frozen=True, slots=True)
class CanonicalLazyBoundaryRegexSource:
    boundary_pattern: str
    boundary_capture_count: int
    token_to_rank: str
    token_capture_ranks: tuple[int, ...]
    token_count: int
    base_token_count: int
    ir: CanonicalLazyBoundaryIRResult
    source_bytes: int
    raw_deflate_bytes: int


def _active_ranks(
    tokens: tuple[bytes | None, ...],
    base_token_count: int,
    applied_merges: int,
) -> tuple[int, ...]:
    active = list(range(base_token_count))
    remaining = applied_merges
    for rank in range(base_token_count, len(tokens)):
        if tokens[rank] is None:
            continue
        if remaining == 0:
            break
        active.append(rank)
        remaining -= 1
    if remaining:
        raise ValueError("adjacency metrics exceed the vocabulary merge count")
    return tuple(active)


def _token_lookup(
    tokens: tuple[bytes | None, ...],
    ranks: tuple[int, ...],
) -> tuple[str, tuple[int, ...]]:
    pairs: list[tuple[bytes, int]] = []
    for rank in ranks:
        token = tokens[rank]
        if token is None:
            raise ValueError(f"reserved token rank {rank} cannot enter lookup")
        pairs.append((token, rank))
    captures: list[int] = []

    def emit(rank: int) -> str:
        captures.append(rank)
        return "()"

    source = render_tagged_regex(
        TaggedFST.from_pairs(pairs).to_regex(),
        escape_byte=lambda byte: f"\\x{byte:02x}",
        emit_tag=emit,
    )
    return source, tuple(captures)


class CanonicalLazyBoundaryRegexCompiler:
    """Connect persistent construction, lazy quotient, elimination, and source."""

    def __init__(
        self,
        tokens: tuple[bytes | None, ...],
        parents: Any,
        *,
        base_token_count: int = 256,
    ) -> None:
        self.tokens = tuple(tokens)
        self.base_token_count = base_token_count
        self.adjacency_compiler = CanonicalAdjacencyCompiler(
            self.tokens,
            parents,
            base_token_count=base_token_count,
        )

    def compile_python(
        self,
        *,
        merge_limit: int | None = None,
        budget: LazyEliminationBudget | None = None,
    ) -> CanonicalLazyBoundaryRegexSource:
        adjacency_result = self.adjacency_compiler.compile(merge_limit=merge_limit)
        quotient = CanonicalLazyQuotientCompiler().compile(adjacency_result.adjacency)
        elimination = CanonicalLazyBoundaryEliminator(budget).lower(
            quotient,
            self.tokens,
        )
        captures = 0

        def emit_boundary() -> str:
            nonlocal captures
            captures += 1
            return "()"

        boundary_source = render_marked_regex(
            elimination.expression,
            escape_byte=lambda byte: f"\\x{byte:02x}",
            emit_boundary=emit_boundary,
        )
        active = _active_ranks(
            self.tokens,
            self.base_token_count,
            adjacency_result.metrics.applied_merges,
        )
        token_source, capture_ranks = _token_lookup(self.tokens, active)
        combined = boundary_source.encode("ascii") + token_source.encode("ascii")
        ir = CanonicalLazyBoundaryIRResult(
            elimination.expression,
            quotient,
            CanonicalLazyBoundaryMetrics(
                adjacency_result.metrics,
                quotient.metrics,
                elimination.metrics,
            ),
        )
        return CanonicalLazyBoundaryRegexSource(
            boundary_source,
            captures,
            token_source,
            capture_ranks,
            len(self.tokens),
            self.base_token_count,
            ir,
            len(combined),
            raw_deflate_size(combined),
        )


__all__ = [
    "CanonicalLazyBoundaryEliminator",
    "CanonicalLazyBoundaryIRResult",
    "CanonicalLazyBoundaryMetrics",
    "CanonicalLazyBoundaryRegexCompiler",
    "CanonicalLazyBoundaryRegexSource",
    "LazyBoundaryEliminationResult",
    "LazyEliminationBudget",
    "LazyEliminationBudgetExceeded",
    "LazyEliminationMetrics",
]
