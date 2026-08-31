"""Opt-in research backends for regular equations.

Nothing in this package is part of the portable Python/ECMAScript artifact
pipeline.  Backends may depend on engine-specific recursive-pattern features.
"""

from .pcre2 import (
    EquationPCRE2BudgetExceeded,
    EquationPCRE2Emitter,
    EquationPCRE2Source,
    render_equation_pcre2,
)

__all__ = [
    "EquationPCRE2BudgetExceeded",
    "EquationPCRE2Emitter",
    "EquationPCRE2Source",
    "render_equation_pcre2",
]
