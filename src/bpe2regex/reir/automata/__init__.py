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
    prune_dead_states,
    prune_unreachable,
    reachable_states,
)
from .canonical import (
    CanonicalTokenDFABudgetExceeded,
    CanonicalTokenDFACompiler,
    CanonicalTokenDFAMetrics,
    CanonicalTokenDFAProgress,
    CanonicalTokenDFAResult,
)
from .canonical_regex import (
    CanonicalTokenRegexCompiler,
    CanonicalTokenRegexIRResult,
    CanonicalTokenRegexMetrics,
    CanonicalTokenRegexSource,
    TokenSymbolLowerer,
    lower_canonical_token_dfa,
    lower_canonical_token_dfa_by_residuals,
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
    LabelLowerer,
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
    "CanonicalTokenDFABudgetExceeded",
    "CanonicalTokenDFACompiler",
    "CanonicalTokenDFAMetrics",
    "CanonicalTokenDFAProgress",
    "CanonicalTokenDFAResult",
    "CanonicalTokenRegexCompiler",
    "CanonicalTokenRegexIRResult",
    "CanonicalTokenRegexMetrics",
    "CanonicalTokenRegexSource",
    "CostGuidedArdenEliminator",
    "DefaultTargetSelector",
    "DefaultTransition",
    "EliminationOrder",
    "EliminationSearchResult",
    "GeneralizedAutomaton",
    "LabelLowerer",
    "LargestSymbolSetDefault",
    "OutputLanguage",
    "SCCEliminationOrder",
    "SemanticAbsorptionResult",
    "SymbolSet",
    "TokenSymbolLowerer",
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
    "lower_canonical_token_dfa",
    "lower_canonical_token_dfa_by_residuals",
    "lower_dfa",
    "minimize_dfa",
    "prune_dead_states",
    "prune_unreachable",
    "reachable_states",
    "strongly_connected_components",
]
