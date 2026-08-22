import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..builder import DEFAULT_BUILDER, RegexBuilder
from ..cost import raw_deflate_size
from ..ops import EPSILON, NEVER, Literal, Op, PureOp
from ..tagged import TAGGED_BUILDER
from ..tagged_source import render_tagged_regex
from .algorithms import coreachable_states, minimize_dfa, prune_dead_states
from .canonical import (
    CanonicalTokenDFACompiler,
    CanonicalTokenDFAMetrics,
    CanonicalTokenDFAProgress,
)
from .elimination import (
    ArdenEliminator,
    SCCEliminationOrder,
    strongly_connected_components,
)
from .ir import DFA
from .labels import SymbolSet


@dataclass(slots=True)
class _TokenTrieNode:
    children: dict[int, _TokenTrieNode] = field(default_factory=dict)
    terminal: bool = False


class TokenSymbolLowerer:
    """Lower finite token-rank labels to prefix-factored byte REIR."""

    def __init__(
        self,
        tokens: Sequence[bytes | None],
        *,
        builder: RegexBuilder = DEFAULT_BUILDER,
    ) -> None:
        self.tokens = tuple(tokens)
        self.builder = builder
        self._cache: dict[int, Op] = {}

    def _lower_node(self, node: _TokenTrieNode) -> Op:
        grouped: dict[Op, list[int]] = {}
        for byte, child in sorted(node.children.items()):
            suffix = self._lower_node(child)
            grouped.setdefault(suffix, []).append(byte)
        alternatives: list[Op] = [EPSILON] if node.terminal else []
        alternatives.extend(
            self.builder.concat(self.builder.charset(bytes_), suffix)
            for suffix, bytes_ in grouped.items()
        )
        return self.builder.alternate(*alternatives)

    def lower(self, symbols: SymbolSet) -> Op:
        if symbols.alphabet_size != len(self.tokens):
            raise ValueError("token symbols and vocabulary have different alphabets")
        cached = self._cache.get(symbols.bits)
        if cached is not None:
            return cached
        if not symbols:
            return NEVER

        root = _TokenTrieNode()
        for rank in symbols:
            token = self.tokens[rank]
            if token is None:
                raise ValueError(f"reserved token rank {rank} cannot be lowered")
            if not token:
                raise ValueError(f"token rank {rank} must not be empty")
            node = root
            for byte in token:
                node = node.children.setdefault(byte, _TokenTrieNode())
            if node.terminal:
                raise ValueError("token byte strings must be unique")
            node.terminal = True
        result = self._lower_node(root)
        self._cache[symbols.bits] = result
        return result

    def __call__(self, symbols: SymbolSet) -> Op:
        return self.lower(symbols)


class _TaggedGeneralizedAutomaton:
    """GNFA whose protected source edges may contain output tags."""

    def __init__(self, states: Sequence[int], source: int, final: int) -> None:
        active = frozenset(states)
        if source not in active or final not in active or source == final:
            raise ValueError("a tagged GNFA requires distinct source and final states")
        self.states = set(active)
        self.source = source
        self.final = final
        self.edges: dict[tuple[int, int], Op] = {}

    def copy(self) -> _TaggedGeneralizedAutomaton:
        result = _TaggedGeneralizedAutomaton(
            tuple(self.states),
            self.source,
            self.final,
        )
        result.edges = self.edges.copy()
        return result

    def add_edge(self, source: int, target: int, label: Op) -> None:
        if source not in self.states or target not in self.states:
            raise ValueError("a tagged GNFA edge endpoint is inactive")
        if label is NEVER:
            return
        previous = self.edges.get((source, target), NEVER)
        self.edges[source, target] = TAGGED_BUILDER.alternate(previous, label)

    def eliminate(self, state: int) -> None:
        if state in (self.source, self.final):
            raise ValueError("the tagged GNFA endpoints cannot be eliminated")
        if state not in self.states:
            raise ValueError("a tagged GNFA state is inactive")

        loop = self.edges.get((state, state), NEVER)
        if not isinstance(loop, PureOp):
            raise TypeError("a tagged GNFA loop must have pure language semantics")
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
                    TAGGED_BUILDER.concat(prefix, closure, suffix),
                )

    @property
    def expression(self) -> Op:
        if self.states != {self.source, self.final}:
            raise ValueError("all tagged GNFA states must be eliminated first")
        return self.edges.get((self.source, self.final), NEVER)

    @property
    def aggregate_expression(self) -> Op:
        return TAGGED_BUILDER.alternate(
            *(self.edges[endpoints] for endpoints in sorted(self.edges))
        )


type TaggedExpressionCost = Callable[[Op], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class CanonicalEliminationSearchResult:
    expression: Op
    order: tuple[int, ...]
    cost: tuple[int, ...]
    explored_candidates: int


@dataclass(slots=True)
class _PartialTaggedElimination:
    graph: _TaggedGeneralizedAutomaton
    order: tuple[int, ...]


def _occurrence_cost(root: Op) -> tuple[int, ...]:
    cache: dict[int, tuple[Op, tuple[int, int]]] = {}

    def visit(op: Op) -> tuple[int, int]:
        known = cache.get(id(op))
        if known is not None and known[0] is op:
            return known[1]
        child_costs = tuple(visit(child) for child in op.operands)
        cost = (
            1 + sum(item[0] for item in child_costs),
            (len(op.value) if isinstance(op, Literal) else 0)
            + sum(item[1] for item in child_costs),
        )
        cache[id(op)] = (op, cost)
        return cost

    return visit(root)


def _python_deflate_cost(root: Op) -> tuple[int, ...]:
    source = render_tagged_regex(
        root,
        escape_byte=lambda byte: f"\\x{byte:02x}",
        emit_tag=lambda _rank: "()",
    )
    return raw_deflate_size(source), len(source.encode("utf-8"))


def _tagged_graph_signature(
    graph: _TaggedGeneralizedAutomaton,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, Op], ...]]:
    return (
        tuple(sorted(graph.states)),
        tuple(
            (source, target, graph.edges[source, target])
            for source, target in sorted(graph.edges)
        ),
    )


@dataclass(frozen=True, slots=True)
class CanonicalTokenRegexMetrics:
    dfa: CanonicalTokenDFAMetrics
    minimized_state_count: int
    minimized_transition_group_count: int
    minimization_seconds: float
    elimination_seconds: float
    elimination_order: tuple[int, ...]
    explored_elimination_candidates: int


@dataclass(frozen=True, slots=True)
class CanonicalTokenRegexIRResult:
    expression: Op
    automaton: DFA[bool]
    metrics: CanonicalTokenRegexMetrics


@dataclass(frozen=True, slots=True)
class CanonicalTokenRegexSource:
    pattern: str
    capture_ranks: tuple[int, ...]
    ir: CanonicalTokenRegexIRResult


def _prepare_canonical_token_graph(
    automaton: DFA[bool],
    tokens: Sequence[bytes | None],
) -> tuple[_TaggedGeneralizedAutomaton, frozenset[int]] | None:
    if automaton.alphabet_size != len(tokens):
        raise ValueError("canonical DFA and vocabulary have different alphabets")
    useful = coreachable_states(automaton)
    if automaton.start not in useful:
        return None

    source = automaton.state_count
    final = source + 1
    graph = _TaggedGeneralizedAutomaton(
        (*sorted(useful), source, final),
        source,
        final,
    )
    lower_symbols = TokenSymbolLowerer(tokens)

    # Only synthetic-source edges emit. A full match therefore validates the
    # complete suffix while its sole participating empty capture marks the end
    # of the suffix's first canonical token.
    for transition in automaton.effective_transitions(automaton.start):
        if transition.target not in useful:
            continue
        for rank in transition.symbols:
            token = tokens[rank]
            if token is None:
                raise ValueError(f"reserved token rank {rank} is reachable")
            graph.add_edge(
                source,
                transition.target,
                TAGGED_BUILDER.concat(
                    TAGGED_BUILDER.literal(token),
                    TAGGED_BUILDER.tag(rank),
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


def lower_canonical_token_dfa(
    automaton: DFA[bool],
    tokens: Sequence[bytes | None],
) -> Op:
    """Lower a canonical-token DFA to a first-boundary tagged byte regex."""
    prepared = _prepare_canonical_token_graph(automaton, tokens)
    if prepared is None:
        return NEVER
    graph, useful = prepared

    order = SCCEliminationOrder().order(automaton, useful)
    for state in order:
        graph.eliminate(state)
    return graph.expression


class CanonicalEliminationOrderSearcher:
    """Beam-search SCC-local orders for the tagged canonical-token GNFA."""

    def __init__(
        self,
        *,
        beam_width: int = 8,
        proxy_cost: TaggedExpressionCost = _occurrence_cost,
        final_cost: TaggedExpressionCost = _python_deflate_cost,
    ) -> None:
        if beam_width <= 0:
            raise ValueError("an elimination beam width must be positive")
        self.beam_width = beam_width
        self.proxy_cost = proxy_cost
        self.final_cost = final_cost

    def _prune(
        self,
        candidates: list[_PartialTaggedElimination],
    ) -> list[_PartialTaggedElimination]:
        unique: dict[
            tuple[tuple[int, ...], tuple[tuple[int, int, Op], ...]],
            _PartialTaggedElimination,
        ] = {}
        for candidate in candidates:
            signature = _tagged_graph_signature(candidate.graph)
            previous = unique.get(signature)
            if previous is None or candidate.order < previous.order:
                unique[signature] = candidate
        return sorted(
            unique.values(),
            key=lambda candidate: (
                self.proxy_cost(candidate.graph.aggregate_expression),
                candidate.order,
            ),
        )[: self.beam_width]

    def search(
        self,
        automaton: DFA[bool],
        tokens: Sequence[bytes | None],
    ) -> CanonicalEliminationSearchResult:
        prepared = _prepare_canonical_token_graph(automaton, tokens)
        if prepared is None:
            return CanonicalEliminationSearchResult(
                NEVER,
                (),
                self.final_cost(NEVER),
                0,
            )
        initial, useful = prepared
        baseline_order = SCCEliminationOrder().order(automaton, useful)
        baseline = initial.copy()
        for state in baseline_order:
            baseline.eliminate(state)

        beam = [_PartialTaggedElimination(initial.copy(), ())]
        explored = 0
        for component in strongly_connected_components(automaton, useful):
            for _ in range(len(component)):
                expanded: list[_PartialTaggedElimination] = []
                for candidate in beam:
                    for state in sorted(component.difference(candidate.order)):
                        graph = candidate.graph.copy()
                        graph.eliminate(state)
                        expanded.append(
                            _PartialTaggedElimination(
                                graph,
                                (*candidate.order, state),
                            )
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
            (index, order, expression, self.final_cost(expression))
            for index, (order, expression) in enumerate(completed)
        )
        _, order, expression, cost = min(
            evaluated,
            key=lambda item: (item[3], item[0]),
        )
        return CanonicalEliminationSearchResult(
            expression,
            order,
            cost,
            explored,
        )


def lower_canonical_token_dfa_by_residuals(
    automaton: DFA[bool],
    tokens: Sequence[bytes | None],
) -> Op:
    """Lower by factoring pure suffix residuals outside first-token tags."""
    if automaton.alphabet_size != len(tokens):
        raise ValueError("canonical DFA and vocabulary have different alphabets")
    useful = coreachable_states(automaton)
    if automaton.start not in useful:
        return NEVER

    lower_symbols = TokenSymbolLowerer(tokens)
    eliminator = ArdenEliminator(label_lowerer=lower_symbols)
    residuals: dict[int, Op] = {}
    alternatives: list[Op] = []
    for transition in automaton.effective_transitions(automaton.start):
        if transition.target not in useful:
            continue
        residual = residuals.get(transition.target)
        if residual is None:
            residual = eliminator.lower_from(automaton, transition.target)
            residuals[transition.target] = residual
        prefixes: list[Op] = []
        for rank in transition.symbols:
            token = tokens[rank]
            if token is None:
                raise ValueError(f"reserved token rank {rank} is reachable")
            prefixes.append(
                TAGGED_BUILDER.concat(
                    TAGGED_BUILDER.literal(token),
                    TAGGED_BUILDER.tag(rank),
                ),
            )
        alternatives.append(
            TAGGED_BUILDER.concat(
                TAGGED_BUILDER.alternate(*prefixes),
                residual,
            )
        )
    return TAGGED_BUILDER.alternate(*alternatives)


class CanonicalTokenRegexCompiler:
    """Connect merge rules through canonical-token DFA and tagged REIR."""

    def __init__(
        self,
        tokens: Sequence[bytes | None],
        parents: Any,
        *,
        base_token_count: int = 256,
    ) -> None:
        self.tokens = tuple(tokens)
        self.dfa_compiler = CanonicalTokenDFACompiler(
            self.tokens,
            parents,
            base_token_count=base_token_count,
        )

    def compile_ir(
        self,
        *,
        merge_limit: int | None = None,
        minimize: bool = True,
        max_states: int | None = None,
        max_transition_groups: int | None = None,
        checkpoint_interval: int = 1_000,
        progress: CanonicalTokenDFAProgress | None = None,
        elimination_beam_width: int | None = None,
    ) -> CanonicalTokenRegexIRResult:
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
        elimination_started = time.perf_counter()
        if elimination_beam_width is None:
            expression = lower_canonical_token_dfa(automaton, self.tokens)
            useful = coreachable_states(automaton)
            elimination_order = SCCEliminationOrder().order(automaton, useful)
            explored_candidates = 0
        else:
            search = CanonicalEliminationOrderSearcher(
                beam_width=elimination_beam_width
            ).search(automaton, self.tokens)
            expression = search.expression
            elimination_order = search.order
            explored_candidates = search.explored_candidates
        elimination_seconds = time.perf_counter() - elimination_started
        return CanonicalTokenRegexIRResult(
            expression,
            automaton,
            CanonicalTokenRegexMetrics(
                dfa_result.metrics,
                automaton.state_count,
                automaton.transition_group_count,
                minimization_seconds,
                elimination_seconds,
                elimination_order,
                explored_candidates,
            ),
        )

    def compile_python(self, **options: Any) -> CanonicalTokenRegexSource:
        ir = self.compile_ir(**options)
        capture_ranks: list[int] = []
        source = render_tagged_regex(
            ir.expression,
            escape_byte=lambda byte: f"\\x{byte:02x}",
            emit_tag=lambda rank: _emit_capture(capture_ranks, rank),
        )
        return CanonicalTokenRegexSource(source, tuple(capture_ranks), ir)


def _emit_capture(capture_ranks: list[int], rank: int) -> str:
    capture_ranks.append(rank)
    return "()"


__all__ = [
    "CanonicalEliminationOrderSearcher",
    "CanonicalEliminationSearchResult",
    "CanonicalTokenRegexCompiler",
    "CanonicalTokenRegexIRResult",
    "CanonicalTokenRegexMetrics",
    "CanonicalTokenRegexSource",
    "TaggedExpressionCost",
    "TokenSymbolLowerer",
    "lower_canonical_token_dfa",
    "lower_canonical_token_dfa_by_residuals",
]
