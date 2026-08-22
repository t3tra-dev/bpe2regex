"""Regular-language IR over bytes plus one observable boundary symbol."""

from dataclasses import dataclass

from .analysis import AnalysisManager, DataFlowAnalysis
from .ops import (
    Alternate,
    CharSet,
    Concat,
    Epsilon,
    LeafOp,
    Literal,
    Never,
    Op,
    Repeat,
)


@dataclass(frozen=True, slots=True)
class Boundary(LeafOp):
    """One symbol in the marked-language alphabet, erased at source lowering.

    ``Boundary`` deliberately has pure regular-language semantics before
    lowering.  It is distinct from every byte, so ordinary union, factoring,
    and state-elimination laws remain valid over the augmented alphabet.
    """


BOUNDARY = Boundary()


@dataclass(frozen=True, slots=True)
class MarkerCount:
    """Inclusive marker-count range over all words in an operation language."""

    can_match: bool
    minimum: int | None
    maximum: int | None

    @property
    def is_exactly_one(self) -> bool:
        return self.can_match and self.minimum == self.maximum == 1


def _sum_optional(values: tuple[int | None, ...]) -> int | None:
    return (
        None
        if any(value is None for value in values)
        else sum(value for value in values if value is not None)
    )


class MarkerCountAnalysis(DataFlowAnalysis[MarkerCount]):
    """Propagate the number of ``Boundary`` symbols on accepting paths."""

    def transfer(
        self,
        op: Op,
        operand_facts: tuple[MarkerCount, ...],
    ) -> MarkerCount:
        match op:
            case Never():
                return MarkerCount(False, None, None)
            case Epsilon() | CharSet() | Literal():
                return MarkerCount(True, 0, 0)
            case Boundary():
                return MarkerCount(True, 1, 1)
            case Concat():
                if not all(fact.can_match for fact in operand_facts):
                    return MarkerCount(False, None, None)
                minimums = tuple(fact.minimum for fact in operand_facts)
                maximums = tuple(fact.maximum for fact in operand_facts)
                return MarkerCount(
                    True,
                    _sum_optional(minimums),
                    _sum_optional(maximums),
                )
            case Alternate():
                matching = tuple(fact for fact in operand_facts if fact.can_match)
                if not matching:
                    return MarkerCount(False, None, None)
                minimums = tuple(
                    fact.minimum for fact in matching if fact.minimum is not None
                )
                maximums = tuple(fact.maximum for fact in matching)
                return MarkerCount(
                    True,
                    min(minimums),
                    None
                    if any(value is None for value in maximums)
                    else max(value for value in maximums if value is not None),
                )
            case Repeat(_, minimum, maximum):
                body = operand_facts[0]
                if not body.can_match:
                    return (
                        MarkerCount(True, 0, 0)
                        if minimum == 0
                        else MarkerCount(False, None, None)
                    )
                assert body.minimum is not None
                lower = body.minimum * minimum
                if maximum == 0 or body.maximum == 0:
                    upper = 0
                elif maximum is None or body.maximum is None:
                    upper = None
                else:
                    upper = body.maximum * maximum
                return MarkerCount(True, lower, upper)
            case _:
                raise TypeError(
                    f"MarkerCountAnalysis has no transfer rule for {type(op).__name__}"
                )


def marker_count(
    expression: Op,
    *,
    analyses: AnalysisManager | None = None,
) -> MarkerCount:
    manager = AnalysisManager() if analyses is None else analyses
    return manager.get(MarkerCountAnalysis, expression)


def verify_single_boundary(
    expression: Op,
    *,
    analyses: AnalysisManager | None = None,
) -> None:
    count = marker_count(expression, analyses=analyses)
    if not count.is_exactly_one:
        raise ValueError(
            "a marked regex must accept with exactly one Boundary; "
            f"observed marker range=({count.minimum}, {count.maximum})"
        )


type MarkedRegex = (
    Never | Epsilon | CharSet | Literal | Concat | Alternate | Repeat | Boundary
)


__all__ = [
    "BOUNDARY",
    "Boundary",
    "MarkedRegex",
    "MarkerCount",
    "MarkerCountAnalysis",
    "marker_count",
    "verify_single_boundary",
]
