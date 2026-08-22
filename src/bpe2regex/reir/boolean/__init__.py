"""Transient Boolean regular-language dialect and core-lowering boundary."""

from .analysis import (
    BooleanRegexProperties,
    BooleanRegexPropertiesAnalysis,
    contains_boolean,
    nullable,
)
from .automata import (
    BooleanDerivativeDFACompiler,
    BooleanDerivativeDFAResult,
    BooleanDerivativeStateBudgetExceeded,
    compile_boolean_dfa,
)
from .builder import (
    BOOLEAN_BUILDER,
    BooleanRegexBuilder,
    complement,
    difference,
    intersect,
)
from .derivative import BooleanDerivativeEngine, boolean_derivative
from .lowering import (
    lower_boolean_ops_to_core,
    lower_boolean_subgraphs,
    lower_boolean_to_core,
)
from .ops import (
    UNIVERSAL,
    BooleanOp,
    Complement,
    Difference,
    Intersect,
    Universal,
)

__all__ = [
    "BOOLEAN_BUILDER",
    "UNIVERSAL",
    "BooleanDerivativeDFACompiler",
    "BooleanDerivativeDFAResult",
    "BooleanDerivativeEngine",
    "BooleanDerivativeStateBudgetExceeded",
    "BooleanOp",
    "BooleanRegexBuilder",
    "BooleanRegexProperties",
    "BooleanRegexPropertiesAnalysis",
    "Complement",
    "Difference",
    "Intersect",
    "Universal",
    "boolean_derivative",
    "compile_boolean_dfa",
    "complement",
    "contains_boolean",
    "difference",
    "intersect",
    "lower_boolean_ops_to_core",
    "lower_boolean_subgraphs",
    "lower_boolean_to_core",
    "nullable",
]
