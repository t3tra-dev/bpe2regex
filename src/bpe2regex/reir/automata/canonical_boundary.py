"""Compile canonical BPE boundaries as a marked regular language."""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ...tagged_fst import TaggedFST
from ..builder import DEFAULT_BUILDER
from ..cost import raw_deflate_size
from ..marked import BOUNDARY, verify_single_boundary
from ..marked_source import render_marked_regex
from ..ops import EPSILON, NEVER, Literal, Op, PureOp
from ..tagged_source import render_tagged_regex
from .algorithms import coreachable_states, minimize_dfa, prune_dead_states
from .canonical import (
    CanonicalTokenDFACompiler,
    CanonicalTokenDFAMetrics,
    CanonicalTokenDFAProgress,
)
from .canonical_regex import TokenSymbolLowerer
from .elimination import SCCEliminationOrder, strongly_connected_components
from .ir import DFA


class BoundaryCostObjective(Enum):
    """Primary objective for marked state-elimination candidate selection."""

    SOURCE = "source"
    DEFLATE = "deflate"
    ARTIFACT = "artifact"


@dataclass(frozen=True, slots=True)
class BoundaryArtifactCost:
    source_bytes: int
    deflate_bytes: int
    artifact_bytes: int
    operation_occurrences: int

    def key(self, objective: BoundaryCostObjective) -> tuple[int, ...]:
        common = (
            self.artifact_bytes,
            self.deflate_bytes,
            self.source_bytes,
            self.operation_occurrences,
        )
        match objective:
            case BoundaryCostObjective.SOURCE:
                return self.source_bytes, *common
            case BoundaryCostObjective.DEFLATE:
                return self.deflate_bytes, *common
            case BoundaryCostObjective.ARTIFACT:
                return common


@dataclass(frozen=True, slots=True)
class CanonicalBoundaryRegexMetrics:
    dfa: CanonicalTokenDFAMetrics
    minimized_state_count: int
    minimized_transition_group_count: int
    minimization_seconds: float
    elimination_seconds: float
    elimination_order: tuple[int, ...]
    explored_elimination_candidates: int


@dataclass(frozen=True, slots=True)
class CanonicalBoundaryRegexIRResult:
    expression: Op
    automaton: DFA[bool]
    metrics: CanonicalBoundaryRegexMetrics


@dataclass(frozen=True, slots=True)
class CanonicalBoundaryRegexSource:
    """Two-regex canonical tokenizer: boundary language plus token lookup."""

    boundary_pattern: str
    boundary_capture_count: int
    token_to_rank: str
    token_capture_ranks: tuple[int, ...]
    token_count: int
    base_token_count: int
    ir: CanonicalBoundaryRegexIRResult
    cost: BoundaryArtifactCost


class _MarkedGeneralizedAutomaton:
    def __init__(self, states: Sequence[int], source: int, final: int) -> None:
        active = frozenset(states)
        if source not in active or final not in active or source == final:
            raise ValueError("a marked GNFA requires distinct source and final states")
        self.states = set(active)
        self.source = source
        self.final = final
        self.edges: dict[tuple[int, int], Op] = {}

    def copy(self) -> _MarkedGeneralizedAutomaton:
        result = _MarkedGeneralizedAutomaton(
            tuple(self.states),
            self.source,
            self.final,
        )
        result.edges = self.edges.copy()
        return result

    def add_edge(self, source: int, target: int, label: Op) -> None:
        if source not in self.states or target not in self.states:
            raise ValueError("a marked GNFA edge endpoint is inactive")
        if label is NEVER:
            return
        previous = self.edges.get((source, target), NEVER)
        self.edges[source, target] = DEFAULT_BUILDER.alternate(previous, label)

    def eliminate(self, state: int) -> None:
        if state in (self.source, self.final):
            raise ValueError("the marked GNFA endpoints cannot be eliminated")
        if state not in self.states:
            raise ValueError("a marked GNFA state is inactive")
        loop = self.edges.get((state, state), NEVER)
        if not isinstance(loop, PureOp):
            raise TypeError("a marked GNFA loop must have pure semantics")
        closure = DEFAULT_BUILDER.repeat(loop, 0, None)
        incoming = tuple(
            (source, label)
            for (source, target), label in self.edges.items()
            if target == state and source != state
        )
        outgoing = tuple(
            (target, label)
            for (source, target), label in self.edges.items()
            if source == state and target != state
        )
        self.edges = {
            endpoints: label
            for endpoints, label in self.edges.items()
            if state not in endpoints
        }
        self.states.remove(state)
        for source, prefix in incoming:
            for target, suffix in outgoing:
                self.add_edge(
                    source,
                    target,
                    DEFAULT_BUILDER.concat(prefix, closure, suffix),
                )

    @property
    def expression(self) -> Op:
        if self.states != {self.source, self.final}:
            raise ValueError("all marked GNFA states must be eliminated first")
        return self.edges.get((self.source, self.final), NEVER)

    @property
    def aggregate_expression(self) -> Op:
        return DEFAULT_BUILDER.alternate(
            *(self.edges[endpoints] for endpoints in sorted(self.edges))
        )


def _prepare_boundary_graph(
    automaton: DFA[bool],
    tokens: Sequence[bytes | None],
) -> tuple[_MarkedGeneralizedAutomaton, frozenset[int]] | None:
    if automaton.alphabet_size != len(tokens):
        raise ValueError("canonical DFA and vocabulary have different alphabets")
    useful = coreachable_states(automaton)
    if automaton.start not in useful:
        return None
    source = automaton.state_count
    final = source + 1
    graph = _MarkedGeneralizedAutomaton(
        (*sorted(useful), source, final),
        source,
        final,
    )
    lower_symbols = TokenSymbolLowerer(tokens)
    for transition in automaton.effective_transitions(automaton.start):
        if transition.target in useful:
            graph.add_edge(
                source,
                transition.target,
                DEFAULT_BUILDER.concat(
                    lower_symbols(transition.symbols),
                    BOUNDARY,
                ),
            )
    for state in sorted(useful):
        for transition in automaton.effective_transitions(state):
            if transition.target in useful:
                graph.add_edge(
                    state,
                    transition.target,
                    lower_symbols(transition.symbols),
                )
        if automaton.outputs[state] is not None:
            graph.add_edge(state, final, EPSILON)
    return graph, useful


def lower_canonical_token_dfa_to_boundary(
    automaton: DFA[bool],
    tokens: Sequence[bytes | None],
) -> Op:
    """Lower a canonical-token DFA to its first-boundary marked language."""
    prepared = _prepare_boundary_graph(automaton, tokens)
    if prepared is None:
        return NEVER
    graph, useful = prepared
    for state in SCCEliminationOrder().order(automaton, useful):
        graph.eliminate(state)
    expression = graph.expression
    verify_single_boundary(expression)
    return expression


def _operation_occurrences(root: Op) -> int:
    cache: dict[int, tuple[Op, int]] = {}

    def visit(op: Op) -> int:
        known = cache.get(id(op))
        if known is not None and known[0] is op:
            return known[1]
        count = 1 + sum(visit(child) for child in op.operands)
        cache[id(op)] = (op, count)
        return count

    return visit(root)


def _encode_uint(value: int) -> bytes:
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _artifact_payload(
    boundary_source: str,
    token_source: str,
    capture_ranks: tuple[int, ...],
    token_count: int,
) -> bytes:
    boundary = boundary_source.encode("ascii")
    lookup = token_source.encode("ascii")
    width = max(1, ((token_count - 1).bit_length() + 7) // 8)
    ranks = b"".join(rank.to_bytes(width, "little") for rank in capture_ranks)
    return b"".join(
        (
            b"B2RB\x01",
            _encode_uint(token_count),
            _encode_uint(len(boundary)),
            boundary,
            _encode_uint(len(lookup)),
            lookup,
            _encode_uint(len(capture_ranks)),
            ranks,
        )
    )


def boundary_artifact_cost(
    expression: Op,
    *,
    token_source: str,
    capture_ranks: tuple[int, ...],
    token_count: int,
) -> BoundaryArtifactCost:
    boundary_source = render_marked_regex(
        expression,
        escape_byte=lambda byte: f"\\x{byte:02x}",
        emit_boundary=lambda: "()",
    )
    combined = boundary_source.encode("ascii") + token_source.encode("ascii")
    artifact = _artifact_payload(
        boundary_source,
        token_source,
        capture_ranks,
        token_count,
    )
    return BoundaryArtifactCost(
        len(combined),
        raw_deflate_size(combined),
        raw_deflate_size(artifact),
        _operation_occurrences(expression),
    )


@dataclass(slots=True)
class _PartialElimination:
    graph: _MarkedGeneralizedAutomaton
    order: tuple[int, ...]


def _graph_signature(
    graph: _MarkedGeneralizedAutomaton,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, Op], ...]]:
    return (
        tuple(sorted(graph.states)),
        tuple(
            (source, target, graph.edges[source, target])
            for source, target in sorted(graph.edges)
        ),
    )


class BoundaryEliminationOrderSearcher:
    """Beam-search marked GNFA orders with complete two-regex costs."""

    def __init__(
        self,
        *,
        token_source: str,
        capture_ranks: tuple[int, ...],
        token_count: int,
        beam_width: int = 8,
        objective: BoundaryCostObjective = BoundaryCostObjective.ARTIFACT,
    ) -> None:
        if beam_width <= 0:
            raise ValueError("an elimination beam width must be positive")
        self.token_source = token_source
        self.capture_ranks = capture_ranks
        self.token_count = token_count
        self.beam_width = beam_width
        self.objective = objective

    def _cost(self, expression: Op) -> BoundaryArtifactCost:
        return boundary_artifact_cost(
            expression,
            token_source=self.token_source,
            capture_ranks=self.capture_ranks,
            token_count=self.token_count,
        )

    def _proxy(self, expression: Op) -> tuple[int, int]:
        literal_bytes = sum(
            len(op.value) for op in _walk_unique(expression) if isinstance(op, Literal)
        )
        return _operation_occurrences(expression), literal_bytes

    def _prune(
        self, candidates: list[_PartialElimination]
    ) -> list[_PartialElimination]:
        unique: dict[
            tuple[tuple[int, ...], tuple[tuple[int, int, Op], ...]],
            _PartialElimination,
        ] = {}
        for candidate in candidates:
            signature = _graph_signature(candidate.graph)
            previous = unique.get(signature)
            if previous is None or candidate.order < previous.order:
                unique[signature] = candidate
        return sorted(
            unique.values(),
            key=lambda candidate: (
                self._proxy(candidate.graph.aggregate_expression),
                candidate.order,
            ),
        )[: self.beam_width]

    def search(
        self,
        automaton: DFA[bool],
        tokens: Sequence[bytes | None],
    ) -> tuple[Op, tuple[int, ...], BoundaryArtifactCost, int]:
        prepared = _prepare_boundary_graph(automaton, tokens)
        if prepared is None:
            cost = self._cost(NEVER)
            return NEVER, (), cost, 0
        initial, useful = prepared
        baseline_order = SCCEliminationOrder().order(automaton, useful)
        baseline = initial.copy()
        for state in baseline_order:
            baseline.eliminate(state)

        beam = [_PartialElimination(initial.copy(), ())]
        explored = 0
        for component in strongly_connected_components(automaton, useful):
            for _ in range(len(component)):
                expanded: list[_PartialElimination] = []
                for candidate in beam:
                    for state in sorted(component.difference(candidate.order)):
                        graph = candidate.graph.copy()
                        graph.eliminate(state)
                        expanded.append(
                            _PartialElimination(graph, (*candidate.order, state))
                        )
                explored += len(expanded)
                beam = self._prune(expanded)

        completed = [(baseline_order, baseline.expression)]
        completed.extend(
            (candidate.order, candidate.graph.expression)
            for candidate in beam
            if candidate.order != baseline_order
        )
        evaluated = tuple(
            (index, order, expression, self._cost(expression))
            for index, (order, expression) in enumerate(completed)
        )
        _, order, expression, cost = min(
            evaluated,
            key=lambda item: (item[3].key(self.objective), item[0]),
        )
        verify_single_boundary(expression)
        return expression, order, cost, explored


def _walk_unique(root: Op) -> tuple[Op, ...]:
    seen: set[int] = set()
    result: list[Op] = []
    pending = [root]
    while pending:
        op = pending.pop()
        if id(op) in seen:
            continue
        seen.add(id(op))
        result.append(op)
        pending.extend(op.operands)
    return tuple(result)


def _active_ranks(
    tokens: Sequence[bytes | None],
    base_token_count: int,
    applied_merges: int,
) -> tuple[int, ...]:
    ranks = list(range(base_token_count))
    remaining = applied_merges
    for rank in range(base_token_count, len(tokens)):
        if tokens[rank] is None:
            continue
        if remaining == 0:
            break
        ranks.append(rank)
        remaining -= 1
    if remaining:
        raise ValueError("DFA metrics refer to more merges than the vocabulary")
    return tuple(ranks)


def _token_lookup_source(
    tokens: Sequence[bytes | None],
    ranks: Sequence[int],
) -> tuple[str, tuple[int, ...]]:
    fst = TaggedFST.from_pairs(
        (tokens[rank], rank) for rank in ranks if tokens[rank] is not None
    )
    captures: list[int] = []
    source = render_tagged_regex(
        fst.to_regex(),
        escape_byte=lambda byte: f"\\x{byte:02x}",
        emit_tag=lambda rank: _capture_rank(captures, rank),
    )
    return source, tuple(captures)


def _capture_rank(captures: list[int], rank: int) -> str:
    captures.append(rank)
    return "()"


class CanonicalBoundaryRegexCompiler:
    """Compile merge rules into a boundary regex and token-rank lookup regex."""

    def __init__(
        self,
        tokens: Sequence[bytes | None],
        parents: Any,
        *,
        base_token_count: int = 256,
    ) -> None:
        self.tokens = tuple(tokens)
        self.base_token_count = base_token_count
        self.dfa_compiler = CanonicalTokenDFACompiler(
            tokens,
            parents,
            base_token_count=base_token_count,
        )

    def compile_python(
        self,
        *,
        merge_limit: int | None = None,
        minimize: bool = True,
        max_states: int | None = None,
        max_transition_groups: int | None = None,
        checkpoint_interval: int = 1_000,
        progress: CanonicalTokenDFAProgress | None = None,
        elimination_beam_width: int | None = None,
        objective: BoundaryCostObjective = BoundaryCostObjective.ARTIFACT,
    ) -> CanonicalBoundaryRegexSource:
        dfa_result = self.dfa_compiler.compile(
            merge_limit=merge_limit,
            max_states=max_states,
            max_transition_groups=max_transition_groups,
            checkpoint_interval=checkpoint_interval,
            progress=progress,
        )
        minimization_started = time.perf_counter()
        automaton = (
            prune_dead_states(minimize_dfa(dfa_result.automaton).automaton).automaton
            if minimize
            else dfa_result.automaton
        )
        minimization_seconds = time.perf_counter() - minimization_started
        active = _active_ranks(
            self.tokens,
            self.base_token_count,
            dfa_result.metrics.applied_merges,
        )
        token_source, capture_ranks = _token_lookup_source(self.tokens, active)

        elimination_started = time.perf_counter()
        if elimination_beam_width is None:
            expression = lower_canonical_token_dfa_to_boundary(
                automaton,
                self.tokens,
            )
            useful = coreachable_states(automaton)
            order = SCCEliminationOrder().order(automaton, useful)
            explored = 0
            cost = boundary_artifact_cost(
                expression,
                token_source=token_source,
                capture_ranks=capture_ranks,
                token_count=len(self.tokens),
            )
        else:
            expression, order, cost, explored = BoundaryEliminationOrderSearcher(
                token_source=token_source,
                capture_ranks=capture_ranks,
                token_count=len(self.tokens),
                beam_width=elimination_beam_width,
                objective=objective,
            ).search(automaton, self.tokens)
        elimination_seconds = time.perf_counter() - elimination_started
        captures = 0

        def emit_boundary() -> str:
            nonlocal captures
            captures += 1
            return "()"

        boundary_source = render_marked_regex(
            expression,
            escape_byte=lambda byte: f"\\x{byte:02x}",
            emit_boundary=emit_boundary,
        )
        ir = CanonicalBoundaryRegexIRResult(
            expression,
            automaton,
            CanonicalBoundaryRegexMetrics(
                dfa_result.metrics,
                automaton.state_count,
                automaton.transition_group_count,
                minimization_seconds,
                elimination_seconds,
                order,
                explored,
            ),
        )
        return CanonicalBoundaryRegexSource(
            boundary_source,
            captures,
            token_source,
            capture_ranks,
            len(self.tokens),
            self.base_token_count,
            ir,
            cost,
        )


__all__ = [
    "BoundaryArtifactCost",
    "BoundaryCostObjective",
    "BoundaryEliminationOrderSearcher",
    "CanonicalBoundaryRegexCompiler",
    "CanonicalBoundaryRegexIRResult",
    "CanonicalBoundaryRegexMetrics",
    "CanonicalBoundaryRegexSource",
    "boundary_artifact_cost",
    "lower_canonical_token_dfa_to_boundary",
]
