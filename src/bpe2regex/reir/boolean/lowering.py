from ..automata.algorithms import minimize_dfa, prune_dead_states
from ..automata.elimination import ArdenEliminator
from ..ops import Op, PureOp
from .analysis import contains_boolean
from .automata import BooleanDerivativeDFACompiler
from .ops import Complement, Difference, Intersect, Universal


def lower_boolean_to_core(
    root: Op,
    *,
    max_states: int | None = 10_000,
    minimize: bool = True,
) -> Op:
    """Eliminate Boolean ops through a derivative DFA and Arden lowering."""
    if not isinstance(root, PureOp):
        raise TypeError("Boolean lowering requires pure regex semantics")
    if not contains_boolean(root):
        return root
    automaton = (
        BooleanDerivativeDFACompiler(max_states=max_states).compile(root).automaton
    )
    if minimize:
        automaton = prune_dead_states(minimize_dfa(automaton).automaton).automaton
    return ArdenEliminator().lower(automaton)


def lower_boolean_subgraphs(
    root: Op,
    *,
    max_states: int | None = 10_000,
    minimize: bool = True,
) -> Op:
    """Lower every maximal pure Boolean subgraph inside possibly tagged REIR."""
    cache: dict[Op, Op] = {}

    def lower(op: Op) -> Op:
        if isinstance(op, PureOp):
            known = cache.get(op)
            if known is not None:
                return known
            result = lower_boolean_to_core(
                op,
                max_states=max_states,
                minimize=minimize,
            )
            cache[op] = result
            return result
        operands = tuple(lower(operand) for operand in op.operands)
        return op if operands == op.operands else op.with_operands(operands)

    return lower(root)


def lower_boolean_ops_to_core(
    root: Op,
    *,
    max_states: int | None = 10_000,
    minimize: bool = True,
) -> Op:
    """Convert innermost Boolean ops while preserving surrounding core structure."""
    cache: dict[Op, Op] = {}
    boolean_types = (Universal, Intersect, Complement, Difference)

    def lower(op: Op) -> Op:
        known = cache.get(op)
        if known is not None:
            return known
        operands = tuple(lower(operand) for operand in op.operands)
        rewritten = op if operands == op.operands else op.with_operands(operands)
        result = (
            lower_boolean_to_core(
                rewritten,
                max_states=max_states,
                minimize=minimize,
            )
            if isinstance(rewritten, boolean_types)
            else rewritten
        )
        cache[op] = result
        return result

    return lower(root)


__all__ = [
    "lower_boolean_ops_to_core",
    "lower_boolean_subgraphs",
    "lower_boolean_to_core",
]
