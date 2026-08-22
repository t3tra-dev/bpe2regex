"""Finite-alphabet automaton IR and semantics-preserving algorithms."""

from .algorithms import (
    AutomatonTransform,
    alphabet_partition,
    coreachable_states,
    equivalence_counterexample,
    equivalent,
    minimize_dfa,
    prune_unreachable,
    reachable_states,
)
from .ir import DFA, Transition
from .labels import SymbolSet

__all__ = [
    "DFA",
    "AutomatonTransform",
    "SymbolSet",
    "Transition",
    "alphabet_partition",
    "coreachable_states",
    "equivalence_counterexample",
    "equivalent",
    "minimize_dfa",
    "prune_unreachable",
    "reachable_states",
]
