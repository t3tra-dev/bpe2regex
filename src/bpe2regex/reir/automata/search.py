from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from ..builder import DEFAULT_BUILDER, RegexBuilder
from ..cost import CostModel
from ..ops import NEVER, Op
from .elimination import (
    ArdenEliminator,
    GeneralizedAutomaton,
    SCCEliminationOrder,
    strongly_connected_components,
)
from .ir import DFA


@dataclass(frozen=True, slots=True)
class EliminationSearchResult[CostT]:
    """The best explored elimination result and its deterministic provenance."""

    expression: Op
    order: tuple[int, ...]
    cost: CostT
    explored_candidates: int


@dataclass(slots=True)
class _PartialElimination:
    graph: GeneralizedAutomaton
    order: tuple[int, ...]


def _graph_signature(
    graph: GeneralizedAutomaton,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, Op], ...]]:
    return (
        tuple(sorted(graph.states)),
        tuple(
            (source, target, graph.edges[source, target])
            for source, target in sorted(graph.edges)
        ),
    )


class CostGuidedArdenEliminator[CostT](ArdenEliminator):
    """Beam-search SCC-local state orders with an arbitrary REIR cost model."""

    def __init__(
        self,
        cost_model: CostModel[CostT],
        *,
        beam_width: int = 16,
        builder: RegexBuilder = DEFAULT_BUILDER,
    ) -> None:
        if beam_width <= 0:
            raise ValueError("an elimination beam width must be positive")
        super().__init__(SCCEliminationOrder(), builder=builder)
        self.cost_model = cost_model
        self.beam_width = beam_width

    def _proxy_key(self, graph: GeneralizedAutomaton) -> tuple[int, ...]:
        expression = (
            graph.expression
            if graph.states == {graph.source, graph.final}
            else graph.aggregate_expression
        )
        return self.cost_model.key(self.cost_model.evaluate(expression))

    def _prune_beam(
        self,
        candidates: Iterable[_PartialElimination],
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
        ranked = sorted(
            unique.values(),
            key=lambda candidate: (self._proxy_key(candidate.graph), candidate.order),
        )
        return ranked[: self.beam_width]

    def search_states[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        final_states: Iterable[int],
    ) -> EliminationSearchResult[CostT]:
        prepared = self.prepare(automaton, final_states)
        if prepared is None:
            cost = self.cost_model.evaluate(NEVER)
            return EliminationSearchResult(NEVER, (), cost, 0)
        initial, useful = prepared

        baseline_order = self.order.order(automaton, useful)
        baseline_graph = initial.copy()
        for state in baseline_order:
            baseline_graph.eliminate(state)

        beam = [_PartialElimination(initial.copy(), ())]
        explored = 0
        for component in strongly_connected_components(automaton, useful):
            for _ in range(len(component)):
                expanded: list[_PartialElimination] = []
                for candidate in beam:
                    remaining = component.difference(candidate.order)
                    for state in sorted(remaining):
                        graph = candidate.graph.copy()
                        graph.eliminate(state)
                        expanded.append(
                            _PartialElimination(graph, (*candidate.order, state))
                        )
                explored += len(expanded)
                beam = self._prune_beam(expanded)

        expressions: list[tuple[tuple[int, ...], Op]] = [
            (baseline_order, baseline_graph.expression)
        ]
        expressions.extend(
            (candidate.order, candidate.graph.expression)
            for candidate in beam
            if candidate.order != baseline_order
        )
        evaluated = tuple(
            (index, order, expression, self.cost_model.evaluate(expression))
            for index, (order, expression) in enumerate(expressions)
        )
        _, order, expression, cost = min(
            evaluated,
            key=lambda item: (self.cost_model.key(item[3]), item[0]),
        )
        return EliminationSearchResult(expression, order, cost, explored)

    def search[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
    ) -> EliminationSearchResult[CostT]:
        return self.search_states(automaton, automaton.accepting_states)

    def lower_states[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        final_states: Iterable[int],
    ) -> Op:
        return self.search_states(automaton, final_states).expression


__all__ = ["CostGuidedArdenEliminator", "EliminationSearchResult"]
