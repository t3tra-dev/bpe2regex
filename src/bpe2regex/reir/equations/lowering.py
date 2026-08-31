"""Explicitly budgeted lowering from equations to the seven core REIR ops."""

import time
from dataclasses import dataclass

from ..automata.canonical_regex import TokenSymbolLowerer
from ..automata.elimination import ArdenEliminator, SCCEliminationOrder
from ..automata.ir import DFA, Transition
from ..automata.labels import SymbolSet
from ..ops import Op
from .analysis import verify_length_guarded
from .ir import RegularEquationIR


@dataclass(frozen=True, slots=True)
class EquationCoreLoweringBudget:
    """Hard limits that keep equation expansion a small-system operation."""

    max_states: int | None = 64
    max_token_checks: int | None = 1_000_000
    max_transition_groups: int | None = 100_000
    max_intermediate_edges: int | None = 100_000
    max_expression_occurrences: int | None = 1_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_states",
            "max_token_checks",
            "max_transition_groups",
            "max_intermediate_edges",
            "max_expression_occurrences",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class EquationCoreLoweringMetrics:
    equation_state_count: int
    active_token_count: int
    materialized_row_count: int
    token_checks: int
    transition_group_count: int
    initial_edge_count: int
    peak_edge_count: int
    eliminated_state_count: int
    elapsed_seconds: float


class EquationCoreLoweringBudgetExceeded(RuntimeError):
    def __init__(self, reason: str, metrics: EquationCoreLoweringMetrics) -> None:
        self.reason = reason
        self.metrics = metrics
        super().__init__(f"equation core-lowering budget exceeded: {reason}")


@dataclass(frozen=True, slots=True)
class EquationCoreLoweringResult:
    expression: Op
    metrics: EquationCoreLoweringMetrics
    state: int | None


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
        cache[id(op)] = op, count
        return count

    return visit(root)


class BudgetedEquationCoreLowerer:
    """Materialize and eliminate equations only when every budget permits it."""

    def __init__(self, budget: EquationCoreLoweringBudget | None = None) -> None:
        self.budget = EquationCoreLoweringBudget() if budget is None else budget

    def lower(
        self,
        ir: RegularEquationIR,
        *,
        state: int | None = None,
    ) -> EquationCoreLoweringResult:
        """Lower the non-empty entry language, or one accepting state residual."""

        verify_length_guarded(ir)
        if state is not None:
            ir.equation(state)
        started = time.perf_counter()
        rows = 0
        checks = 0
        groups = 0
        initial_edges = 0
        peak_edges = 0
        eliminated = 0

        def metrics() -> EquationCoreLoweringMetrics:
            return EquationCoreLoweringMetrics(
                ir.state_count,
                len(ir.active_tokens),
                rows,
                checks,
                groups,
                initial_edges,
                peak_edges,
                eliminated,
                time.perf_counter() - started,
            )

        def require(condition: bool, reason: str) -> None:
            if not condition:
                raise EquationCoreLoweringBudgetExceeded(reason, metrics())

        if self.budget.max_states is not None:
            require(ir.state_count <= self.budget.max_states, "equation state count")
        projected_checks = ir.state_count * len(ir.active_tokens)
        if self.budget.max_token_checks is not None:
            require(projected_checks <= self.budget.max_token_checks, "token checks")

        materialized: list[tuple[Transition, ...]] = []
        for source in range(ir.state_count):
            denied = ir.denial_selector(source)
            target_bits: dict[int, int] = {}
            for token in ir.active_tokens:
                checks += 1
                if ir.selector_dag.contains(denied, token):
                    continue
                target = ir.token_target(token)
                target_bits[target] = target_bits.get(target, 0) | (1 << token)
            row = tuple(
                Transition(SymbolSet(len(ir.tokens), bits), target)
                for target, bits in sorted(target_bits.items())
            )
            materialized.append(row)
            rows += 1
            groups += len(row)
            if self.budget.max_transition_groups is not None:
                require(
                    groups <= self.budget.max_transition_groups,
                    "transition group count",
                )

        outputs: list[bool | None] = [True] * ir.state_count
        dfa_rows = list(materialized)
        start = ir.start if state is None else state
        if state is None:
            start = len(outputs)
            outputs.append(None)
            dfa_rows.append(materialized[ir.start])
            groups += len(materialized[ir.start])
            if self.budget.max_transition_groups is not None:
                require(
                    groups <= self.budget.max_transition_groups,
                    "entry transition group count",
                )

        automaton = DFA(
            len(ir.tokens),
            start,
            tuple(outputs),
            tuple(dfa_rows),
        )
        eliminator = ArdenEliminator(label_lowerer=TokenSymbolLowerer(ir.tokens))
        prepared = eliminator.prepare_from(
            automaton,
            start,
            range(ir.state_count),
        )
        if prepared is None:
            raise AssertionError("a canonical equation entry must reach acceptance")
        graph, useful = prepared
        initial_edges = len(graph.edges)
        peak_edges = initial_edges
        if self.budget.max_intermediate_edges is not None:
            require(initial_edges <= self.budget.max_intermediate_edges, "initial edge count")

        order = SCCEliminationOrder().order(automaton, useful)
        for eliminated_state in order:
            graph.eliminate(eliminated_state)
            eliminated += 1
            peak_edges = max(peak_edges, len(graph.edges))
            if self.budget.max_intermediate_edges is not None:
                require(
                    len(graph.edges) <= self.budget.max_intermediate_edges,
                    "intermediate edge count",
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

        return EquationCoreLoweringResult(graph.expression, metrics(), state)


def lower_regular_equations_to_core(
    ir: RegularEquationIR,
    *,
    state: int | None = None,
    budget: EquationCoreLoweringBudget | None = None,
) -> Op:
    return BudgetedEquationCoreLowerer(budget).lower(ir, state=state).expression


__all__ = [
    "BudgetedEquationCoreLowerer",
    "EquationCoreLoweringBudget",
    "EquationCoreLoweringBudgetExceeded",
    "EquationCoreLoweringMetrics",
    "EquationCoreLoweringResult",
    "lower_regular_equations_to_core",
]
