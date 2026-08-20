from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from .ops import (
    Alternate,
    CharSet,
    Concat,
    Epsilon,
    Literal,
    Never,
    Op,
    PureOp,
    Repeat,
)


class DataFlowAnalysis[FactT](ABC):
    """A bottom-up analysis whose transfer function produces one fact per op."""

    @abstractmethod
    def transfer(self, op: Op, operand_facts: tuple[FactT, ...]) -> FactT:
        """Compute an operation fact from already-computed operand facts."""


type AnalysisType = type[DataFlowAnalysis[Any]]


class AnalysisManager:
    """Lazily evaluate and cache data-flow analyses over immutable REIR."""

    def __init__(self) -> None:
        self._facts: dict[AnalysisType, dict[int, tuple[Op, Any]]] = {}

    def get[FactT](
        self,
        analysis_type: type[DataFlowAnalysis[FactT]],
        root: Op,
    ) -> FactT:
        analysis = analysis_type()
        cache = self._facts.setdefault(analysis_type, {})
        active: set[int] = set()

        def evaluate(op: Op) -> FactT:
            identity = id(op)
            known = cache.get(identity)
            if known is not None and known[0] is op:
                return cast(FactT, known[1])
            if identity in active:
                raise ValueError("REIR analyses require an acyclic operation graph")
            active.add(identity)
            try:
                op.verify()
                operand_facts = tuple(evaluate(operand) for operand in op.operands)
                fact = analysis.transfer(op, operand_facts)
                cache[identity] = (op, fact)
                return fact
            finally:
                active.remove(identity)

        return evaluate(root)

    def invalidate(self, analysis_types: Iterable[AnalysisType] | None = None) -> None:
        if analysis_types is None:
            self._facts.clear()
            return
        for analysis_type in analysis_types:
            self._facts.pop(analysis_type, None)

    def invalidate_except(self, preserved: Iterable[AnalysisType]) -> None:
        preserved_types = frozenset(preserved)
        for analysis_type in tuple(self._facts):
            if analysis_type not in preserved_types:
                del self._facts[analysis_type]


class PurityAnalysis(DataFlowAnalysis[bool]):
    """Prove that an operation graph has pure regular-language semantics."""

    def transfer(self, op: Op, operand_facts: tuple[bool, ...]) -> bool:
        return isinstance(op, PureOp) and all(operand_facts)


@dataclass(frozen=True, slots=True)
class RegexProperties:
    """Language and structural facts propagated through pure regex REIR."""

    can_match: bool
    nullable: bool
    min_width: int | None
    max_width: int | None
    first_symbols: frozenset[int]
    last_symbols: frozenset[int]
    operation_count: int
    literal_bytes: int


class RegexPropertiesAnalysis(DataFlowAnalysis[RegexProperties]):
    """Infer nullability, first/last sets, widths, and structural costs."""

    def transfer(
        self,
        op: Op,
        operand_facts: tuple[RegexProperties, ...],
    ) -> RegexProperties:
        operation_count = 1 + sum(fact.operation_count for fact in operand_facts)
        literal_bytes = sum(fact.literal_bytes for fact in operand_facts)

        match op:
            case Never():
                return RegexProperties(
                    False, False, None, None, frozenset(), frozenset(), 1, 0
                )
            case Epsilon():
                return RegexProperties(True, True, 0, 0, frozenset(), frozenset(), 1, 0)
            case CharSet():
                symbols = op.symbols
                return RegexProperties(True, False, 1, 1, symbols, symbols, 1, 0)
            case Literal(value):
                width = len(value)
                return RegexProperties(
                    True,
                    False,
                    width,
                    width,
                    frozenset((value[0],)),
                    frozenset((value[-1],)),
                    1,
                    width,
                )
            case Concat():
                if not all(fact.can_match for fact in operand_facts):
                    return RegexProperties(
                        False,
                        False,
                        None,
                        None,
                        frozenset(),
                        frozenset(),
                        operation_count,
                        literal_bytes,
                    )
                first: set[int] = set()
                for fact in operand_facts:
                    first.update(fact.first_symbols)
                    if not fact.nullable:
                        break
                last: set[int] = set()
                for fact in reversed(operand_facts):
                    last.update(fact.last_symbols)
                    if not fact.nullable:
                        break
                minimum = sum(
                    fact.min_width
                    for fact in operand_facts
                    if fact.min_width is not None
                )
                maximum = (
                    None
                    if any(fact.max_width is None for fact in operand_facts)
                    else sum(
                        fact.max_width
                        for fact in operand_facts
                        if fact.max_width is not None
                    )
                )
                return RegexProperties(
                    True,
                    all(fact.nullable for fact in operand_facts),
                    minimum,
                    maximum,
                    frozenset(first),
                    frozenset(last),
                    operation_count,
                    literal_bytes,
                )
            case Alternate():
                matching = [fact for fact in operand_facts if fact.can_match]
                if not matching:
                    return RegexProperties(
                        False,
                        False,
                        None,
                        None,
                        frozenset(),
                        frozenset(),
                        operation_count,
                        literal_bytes,
                    )
                minimum = min(
                    fact.min_width for fact in matching if fact.min_width is not None
                )
                maximum = (
                    None
                    if any(fact.max_width is None for fact in matching)
                    else max(
                        fact.max_width
                        for fact in matching
                        if fact.max_width is not None
                    )
                )
                return RegexProperties(
                    True,
                    any(fact.nullable for fact in matching),
                    minimum,
                    maximum,
                    frozenset().union(*(fact.first_symbols for fact in matching)),
                    frozenset().union(*(fact.last_symbols for fact in matching)),
                    operation_count,
                    literal_bytes,
                )
            case Repeat(_, minimum, maximum):
                body_fact = operand_facts[0]
                if not body_fact.can_match:
                    if minimum == 0:
                        return RegexProperties(
                            True,
                            True,
                            0,
                            0,
                            frozenset(),
                            frozenset(),
                            operation_count,
                            literal_bytes,
                        )
                    return RegexProperties(
                        False,
                        False,
                        None,
                        None,
                        frozenset(),
                        frozenset(),
                        operation_count,
                        literal_bytes,
                    )
                assert body_fact.min_width is not None
                min_width = body_fact.min_width * minimum
                if maximum == 0 or body_fact.max_width == 0:
                    max_width = 0
                elif maximum is None or body_fact.max_width is None:
                    max_width = None
                else:
                    max_width = body_fact.max_width * maximum
                symbols = frozenset() if maximum == 0 else body_fact.first_symbols
                last_symbols = frozenset() if maximum == 0 else body_fact.last_symbols
                return RegexProperties(
                    True,
                    minimum == 0 or body_fact.nullable,
                    min_width,
                    max_width,
                    symbols,
                    last_symbols,
                    operation_count,
                    literal_bytes,
                )
            case _:
                raise TypeError(
                    f"RegexPropertiesAnalysis has no transfer rule for "
                    f"{type(op).__name__}"
                )


__all__ = [
    "AnalysisManager",
    "AnalysisType",
    "DataFlowAnalysis",
    "PurityAnalysis",
    "RegexProperties",
    "RegexPropertiesAnalysis",
]
