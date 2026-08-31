"""Static analyses for the regular-equation dialect."""

from dataclasses import dataclass

from .ir import (
    EquationAlternate,
    EquationConcat,
    EquationDifference,
    EquationEpsilon,
    EquationExpr,
    EquationLiteral,
    EquationNever,
    EquationRef,
    EquationTokenSelectorRef,
    EquationUniversal,
    RegularEquationIR,
)


@dataclass(frozen=True, slots=True)
class ReferenceGuard:
    """The guaranteed consumed prefix at one explicit or selector reference."""

    equation: int | None
    target: int | None
    consumed_bytes: int
    selector: int | None = None


@dataclass(frozen=True, slots=True)
class LengthGuardednessReport:
    references: tuple[ReferenceGuard, ...]
    violations: tuple[ReferenceGuard, ...]

    @property
    def is_guarded(self) -> bool:
        return not self.violations

    @property
    def minimum_guard_bytes(self) -> int | None:
        return (
            min(reference.consumed_bytes for reference in self.references)
            if self.references
            else None
        )


class UnguardedEquationError(ValueError):
    def __init__(self, report: LengthGuardednessReport) -> None:
        self.report = report
        owners = ", ".join(
            "entry" if item.equation is None else f"Q{item.equation}"
            for item in report.violations[:4]
        )
        suffix = "" if len(report.violations) <= 4 else ", ..."
        super().__init__(
            "regular-equation references must consume at least one byte; "
            f"unguarded references in {owners}{suffix}"
        )


def _minimum_length(expression: EquationExpr, token_minimum: int) -> int | None:
    """Return a conservative accepted-word lower bound; ``None`` means empty."""
    match expression:
        case EquationNever():
            return None
        case EquationEpsilon() | EquationUniversal() | EquationRef():
            return 0
        case EquationLiteral(value):
            return len(value)
        case EquationTokenSelectorRef():
            return token_minimum
        case EquationConcat(parts):
            lengths = tuple(_minimum_length(part, token_minimum) for part in parts)
            if any(length is None for length in lengths):
                return None
            return sum(length for length in lengths if length is not None)
        case EquationAlternate(alternatives):
            lengths = tuple(
                length
                for branch in alternatives
                if (length := _minimum_length(branch, token_minimum)) is not None
            )
            return min(lengths) if lengths else None
        case EquationDifference(left, _):
            # Difference is a subset of its left operand, so the left lower
            # bound remains sound even when it is not exact.
            return _minimum_length(left, token_minimum)
    raise TypeError(f"unsupported equation expression: {type(expression).__name__}")


def analyze_length_guardedness(ir: RegularEquationIR) -> LengthGuardednessReport:
    """Prove that every recursive reference is preceded by a non-empty word.

    Selector references are checked symbolically: module verification already
    proves that every active token is non-empty, so no denied-token set needs
    to be expanded for this analysis.
    """

    token_minimum = min(len(ir.tokens[token] or b"") for token in ir.active_tokens)
    if token_minimum <= 0:
        raise ValueError("length guardedness requires non-empty active tokens")
    references: list[ReferenceGuard] = []

    def visit(expression: EquationExpr, consumed: int, owner: int | None) -> None:
        match expression:
            case EquationRef(variable):
                references.append(ReferenceGuard(owner, variable, consumed))
            case EquationTokenSelectorRef(selector):
                selected = ir.selector_dag.selectors[selector]
                if selected.all_tokens or selected.cone_ids:
                    references.append(
                        ReferenceGuard(
                            owner,
                            None,
                            consumed + token_minimum,
                            selector,
                        )
                    )
            case EquationConcat(parts):
                prefix = consumed
                reachable = True
                for part in parts:
                    if reachable:
                        visit(part, prefix, owner)
                    length = _minimum_length(part, token_minimum)
                    if length is None:
                        reachable = False
                    else:
                        prefix += length
            case EquationAlternate(alternatives):
                for branch in alternatives:
                    visit(branch, consumed, owner)
            case EquationDifference(left, right):
                visit(left, consumed, owner)
                visit(right, consumed, owner)
            case (
                EquationNever()
                | EquationEpsilon()
                | EquationUniversal()
                | EquationLiteral()
            ):
                return
            case _:
                raise TypeError(
                    f"unsupported equation expression: {type(expression).__name__}"
                )

    visit(ir.entry, 0, None)
    for equation in ir.equations:
        visit(equation.body, 0, equation.variable)
    result = tuple(references)
    return LengthGuardednessReport(
        result,
        tuple(reference for reference in result if reference.consumed_bytes <= 0),
    )


def verify_length_guarded(ir: RegularEquationIR) -> LengthGuardednessReport:
    report = analyze_length_guardedness(ir)
    if not report.is_guarded:
        raise UnguardedEquationError(report)
    return report


__all__ = [
    "LengthGuardednessReport",
    "ReferenceGuard",
    "UnguardedEquationError",
    "analyze_length_guardedness",
    "verify_length_guarded",
]
