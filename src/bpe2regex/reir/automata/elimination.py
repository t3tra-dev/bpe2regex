from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass

from ..builder import DEFAULT_BUILDER, RegexBuilder
from ..ops import EPSILON, NEVER, Op, PureOp
from .ir import DFA
from .labels import SymbolSet

type LabelLowerer = Callable[[SymbolSet], Op]


@dataclass(frozen=True, slots=True)
class OutputLanguage[OutputT: Hashable]:
    """One observable DFA output and the language producing it."""

    output: OutputT
    expression: Op


class EliminationOrder(ABC):
    """Extensible policy for ordering useful DFA states."""

    @abstractmethod
    def order[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        states: frozenset[int],
    ) -> tuple[int, ...]:
        """Return every requested state exactly once."""


def _successors[OutputT: Hashable](
    automaton: DFA[OutputT],
    states: frozenset[int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            sorted(
                {
                    transition.target
                    for transition in automaton.effective_transitions(state)
                    if transition.target in states
                }
            )
        )
        for state in range(automaton.state_count)
    )


def strongly_connected_components[OutputT: Hashable](
    automaton: DFA[OutputT],
    states: Iterable[int] | None = None,
) -> tuple[frozenset[int], ...]:
    """Return deterministic SCCs in reverse topological order without recursion."""
    selected = (
        frozenset(range(automaton.state_count)) if states is None else frozenset(states)
    )
    if any(not 0 <= state < automaton.state_count for state in selected):
        raise ValueError("an SCC state is out of range")
    successors = _successors(automaton, selected)
    predecessors: list[list[int]] = [[] for _ in range(automaton.state_count)]
    for source in selected:
        for target in successors[source]:
            predecessors[target].append(source)
    for row in predecessors:
        row.sort()

    visited: set[int] = set()
    finish_order: list[int] = []
    for root in sorted(selected):
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            state, successor_index = stack[-1]
            if successor_index < len(successors[state]):
                target = successors[state][successor_index]
                stack[-1] = state, successor_index + 1
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


class SCCEliminationOrder(EliminationOrder):
    """Eliminate downstream SCCs first, using state ids within each SCC."""

    def order[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        states: frozenset[int],
    ) -> tuple[int, ...]:
        return tuple(
            state
            for component in strongly_connected_components(automaton, states)
            for state in sorted(component)
        )


class GeneralizedAutomaton:
    """Mutable GNFA used while applying Arden state elimination."""

    def __init__(
        self,
        states: Iterable[int],
        source: int,
        final: int,
        *,
        builder: RegexBuilder = DEFAULT_BUILDER,
    ) -> None:
        active = frozenset(states)
        if source not in active or final not in active or source == final:
            raise ValueError("a GNFA requires distinct active source and final states")
        self.states = set(active)
        self.source = source
        self.final = final
        self.builder = builder
        self.edges: dict[tuple[int, int], Op] = {}

    def copy(self) -> GeneralizedAutomaton:
        result = GeneralizedAutomaton(
            self.states,
            self.source,
            self.final,
            builder=self.builder,
        )
        result.edges = self.edges.copy()
        return result

    def add_edge(self, source: int, target: int, label: Op) -> None:
        if source not in self.states or target not in self.states:
            raise ValueError("a GNFA edge endpoint is inactive")
        if not isinstance(label, PureOp):
            raise TypeError("a GNFA edge label must be pure REIR")
        if label is NEVER:
            return
        previous = self.edges.get((source, target), NEVER)
        self.edges[source, target] = self.builder.alternate(previous, label)

    def eliminate(self, state: int) -> None:
        if state in (self.source, self.final):
            raise ValueError("the synthetic GNFA endpoints cannot be eliminated")
        if state not in self.states:
            raise ValueError("a GNFA state is inactive")

        loop = self.edges.get((state, state), NEVER)
        closure = self.builder.repeat(loop, 0, None)
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
                    self.builder.concat(prefix, closure, suffix),
                )

    @property
    def expression(self) -> Op:
        if self.states != {self.source, self.final}:
            raise ValueError("all non-endpoint GNFA states must be eliminated first")
        return self.edges.get((self.source, self.final), NEVER)

    @property
    def aggregate_expression(self) -> Op:
        """Return a deterministic proxy expression covering all active edges."""
        return self.builder.alternate(
            *(self.edges[endpoints] for endpoints in sorted(self.edges))
        )


def _reachable_from[OutputT: Hashable](
    automaton: DFA[OutputT],
    start: int,
) -> set[int]:
    if not 0 <= start < automaton.state_count:
        raise ValueError("an Arden start state is out of range")
    reachable = {start}
    pending = deque((start,))
    while pending:
        state = pending.popleft()
        for transition in automaton.effective_transitions(state):
            if transition.target not in reachable:
                reachable.add(transition.target)
                pending.append(transition.target)
    return reachable


def _coreachable_to[OutputT: Hashable](
    automaton: DFA[OutputT],
    final_states: frozenset[int],
) -> set[int]:
    predecessors: list[set[int]] = [set() for _ in range(automaton.state_count)]
    for source in range(automaton.state_count):
        for transition in automaton.effective_transitions(source):
            predecessors[transition.target].add(source)
    result = set(final_states)
    pending = deque(sorted(final_states))
    while pending:
        target = pending.popleft()
        for source in sorted(predecessors[target]):
            if source not in result:
                result.add(source)
                pending.append(source)
    return result


class ArdenEliminator:
    """Lower byte DFAs to pure REIR with SCC-aware Arden elimination."""

    def __init__(
        self,
        order: EliminationOrder | None = None,
        *,
        builder: RegexBuilder = DEFAULT_BUILDER,
        label_lowerer: LabelLowerer | None = None,
    ) -> None:
        self.order = SCCEliminationOrder() if order is None else order
        self.builder = builder
        self.label_lowerer = label_lowerer

    def _useful_states[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        start: int,
        final_states: frozenset[int],
    ) -> frozenset[int]:
        if any(not 0 <= state < automaton.state_count for state in final_states):
            raise ValueError("an Arden final state is out of range")
        return frozenset(
            _reachable_from(automaton, start) & _coreachable_to(automaton, final_states)
        )

    def prepare_from[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        start: int,
        final_states: Iterable[int],
    ) -> tuple[GeneralizedAutomaton, frozenset[int]] | None:
        if self.label_lowerer is None and automaton.alphabet_size != 256:
            raise ValueError("only byte-alphabet DFAs can lower to pure REIR")
        finals = frozenset(final_states)
        useful = self._useful_states(automaton, start, finals)
        if start not in useful:
            return None

        source = automaton.state_count
        final = source + 1
        graph = GeneralizedAutomaton(
            (*useful, source, final),
            source,
            final,
            builder=self.builder,
        )
        graph.add_edge(source, start, EPSILON)
        for state in sorted(useful):
            for transition in automaton.effective_transitions(state):
                if transition.target in useful:
                    label = (
                        transition.symbols.to_reir()
                        if self.label_lowerer is None
                        else self.label_lowerer(transition.symbols)
                    )
                    graph.add_edge(state, transition.target, label)
            if state in finals:
                graph.add_edge(state, final, EPSILON)
        return graph, useful

    def prepare[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        final_states: Iterable[int],
    ) -> tuple[GeneralizedAutomaton, frozenset[int]] | None:
        return self.prepare_from(automaton, automaton.start, final_states)

    def lower_from_states[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        start: int,
        final_states: Iterable[int],
    ) -> Op:
        prepared = self.prepare_from(automaton, start, final_states)
        if prepared is None:
            return NEVER
        graph, useful = prepared
        order = self.order.order(automaton, useful)
        if len(order) != len(useful) or frozenset(order) != useful:
            raise ValueError(
                "an elimination order must contain every useful state once"
            )
        for state in order:
            graph.eliminate(state)
        return graph.expression

    def lower_states[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        final_states: Iterable[int],
    ) -> Op:
        return self.lower_from_states(automaton, automaton.start, final_states)

    def lower_from[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        start: int,
    ) -> Op:
        return self.lower_from_states(automaton, start, automaton.accepting_states)

    def lower[OutputT: Hashable](self, automaton: DFA[OutputT]) -> Op:
        return self.lower_states(automaton, automaton.accepting_states)

    def lower_output[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        output: OutputT,
    ) -> Op:
        if output is None:
            raise ValueError("None is the reserved rejecting output")
        return self.lower_states(
            automaton,
            (
                state
                for state, candidate in enumerate(automaton.outputs)
                if candidate == output
            ),
        )

    def lower_outputs[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
    ) -> tuple[OutputLanguage[OutputT], ...]:
        outputs: list[OutputT] = []
        seen: set[OutputT] = set()
        for output in automaton.outputs:
            if output is not None and output not in seen:
                seen.add(output)
                outputs.append(output)
        return tuple(
            OutputLanguage(output, self.lower_output(automaton, output))
            for output in outputs
        )


def lower_dfa[OutputT: Hashable](automaton: DFA[OutputT]) -> Op:
    return ArdenEliminator().lower(automaton)


__all__ = [
    "ArdenEliminator",
    "EliminationOrder",
    "GeneralizedAutomaton",
    "LabelLowerer",
    "OutputLanguage",
    "SCCEliminationOrder",
    "lower_dfa",
    "strongly_connected_components",
]
