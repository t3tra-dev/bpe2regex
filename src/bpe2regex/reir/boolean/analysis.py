from dataclasses import dataclass

from ..ops import (
    BYTE_ALPHABET_SIZE,
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
from .ops import Complement, Difference, Intersect, Universal


@dataclass(frozen=True, slots=True)
class BooleanRegexProperties:
    """Exact nullability plus structural facts for Boolean REIR."""

    nullable: bool
    first_symbols: frozenset[int]
    operation_count: int
    literal_bytes: int
    boolean_operation_count: int


class BooleanRegexPropertiesAnalysis:
    """Memoized bottom-up analysis over core and Boolean pure operations."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[Op, BooleanRegexProperties]] = {}
        self._active: set[int] = set()

    def get(self, root: Op) -> BooleanRegexProperties:
        if not isinstance(root, PureOp):
            raise TypeError("Boolean regex analysis requires pure regex semantics")
        identity = id(root)
        known = self._cache.get(identity)
        if known is not None and known[0] is root:
            return known[1]
        if identity in self._active:
            raise ValueError("Boolean REIR analyses require an acyclic graph")
        self._active.add(identity)
        try:
            operands = tuple(self.get(operand) for operand in root.operands)
            operation_count = 1 + sum(item.operation_count for item in operands)
            literal_bytes = sum(item.literal_bytes for item in operands)
            boolean_count = sum(item.boolean_operation_count for item in operands)
            match root:
                case Never() | CharSet() | Literal():
                    nullable = False
                    if isinstance(root, CharSet):
                        first_symbols = root.symbols
                    elif isinstance(root, Literal):
                        first_symbols = frozenset((root.value[0],))
                    else:
                        first_symbols = frozenset()
                case Epsilon() | Universal():
                    nullable = True
                    first_symbols = (
                        frozenset(range(BYTE_ALPHABET_SIZE))
                        if isinstance(root, Universal)
                        else frozenset()
                    )
                case Concat():
                    nullable = all(item.nullable for item in operands)
                    first: set[int] = set()
                    for item in operands:
                        first.update(item.first_symbols)
                        if not item.nullable:
                            break
                    first_symbols = frozenset(first)
                case Alternate():
                    nullable = any(item.nullable for item in operands)
                    first_symbols = frozenset().union(
                        *(item.first_symbols for item in operands)
                    )
                case Repeat(_, minimum, _):
                    nullable = minimum == 0 or operands[0].nullable
                    first_symbols = operands[0].first_symbols
                case Intersect():
                    nullable = all(item.nullable for item in operands)
                    first_symbols = frozenset.intersection(
                        *(item.first_symbols for item in operands)
                    )
                    boolean_count += 1
                case Complement():
                    nullable = not operands[0].nullable
                    first_symbols = frozenset(range(BYTE_ALPHABET_SIZE))
                    boolean_count += 1
                case Difference():
                    nullable = operands[0].nullable and not operands[1].nullable
                    first_symbols = operands[0].first_symbols
                    boolean_count += 1
                case _:
                    raise TypeError(
                        f"BooleanRegexPropertiesAnalysis has no rule for "
                        f"{type(root).__name__}"
                    )
            result = BooleanRegexProperties(
                nullable,
                first_symbols,
                operation_count,
                literal_bytes + (len(root.value) if isinstance(root, Literal) else 0),
                boolean_count + (1 if isinstance(root, Universal) else 0),
            )
            self._cache[identity] = root, result
            return result
        finally:
            self._active.remove(identity)


def nullable(root: Op) -> bool:
    return BooleanRegexPropertiesAnalysis().get(root).nullable


def contains_boolean(root: Op) -> bool:
    """Return whether a possibly tagged operation graph contains this dialect."""
    seen: set[int] = set()

    def visit(op: Op) -> bool:
        if isinstance(op, (Universal, Intersect, Complement, Difference)):
            return True
        identity = id(op)
        if identity in seen:
            return False
        seen.add(identity)
        return any(visit(operand) for operand in op.operands)

    return visit(root)


__all__ = [
    "BooleanRegexProperties",
    "BooleanRegexPropertiesAnalysis",
    "contains_boolean",
    "nullable",
]
