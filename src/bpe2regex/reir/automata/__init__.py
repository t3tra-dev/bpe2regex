"""Finite-alphabet automaton IR and semantics-preserving algorithms."""

from .absorption import (
    AutomatonSemanticAbsorber,
    SemanticAbsorptionResult,
    absorb_acceptance_union,
    acceptance_included,
    acceptance_inclusion_counterexample,
)
from .algorithms import (
    AutomatonTransform,
    alphabet_partition,
    combined_alphabet_partition,
    coreachable_states,
    equivalence_counterexample,
    equivalent,
    minimize_dfa,
    prune_unreachable,
    reachable_states,
)
from .defaults import (
    DefaultTargetSelector,
    LargestSymbolSetDefault,
    encode_default_transitions,
    expand_default_transitions,
)
from .elimination import (
    ArdenEliminator,
    EliminationOrder,
    GeneralizedAutomaton,
    OutputLanguage,
    SCCEliminationOrder,
    lower_dfa,
    strongly_connected_components,
)
from .ir import DFA, DefaultTransition, Transition
from .labels import SymbolSet
from .pipeline import AcceptanceAutomataCompiler, AutomatonCompilationResult
from .search import CostGuidedArdenEliminator, EliminationSearchResult

__all__ = [
    "DFA",
    "AcceptanceAutomataCompiler",
    "ArdenEliminator",
    "AutomatonCompilationResult",
    "AutomatonSemanticAbsorber",
    "AutomatonTransform",
    "CostGuidedArdenEliminator",
    "DefaultTargetSelector",
    "DefaultTransition",
    "EliminationOrder",
    "EliminationSearchResult",
    "GeneralizedAutomaton",
    "LargestSymbolSetDefault",
    "OutputLanguage",
    "SCCEliminationOrder",
    "SemanticAbsorptionResult",
    "SymbolSet",
    "Transition",
    "absorb_acceptance_union",
    "acceptance_included",
    "acceptance_inclusion_counterexample",
    "alphabet_partition",
    "combined_alphabet_partition",
    "coreachable_states",
    "encode_default_transitions",
    "equivalence_counterexample",
    "equivalent",
    "expand_default_transitions",
    "lower_dfa",
    "minimize_dfa",
    "prune_unreachable",
    "reachable_states",
    "strongly_connected_components",
]
