from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

BYTE_ALPHABET_SIZE = 256
BYTE_ALPHABET_MASK = (1 << BYTE_ALPHABET_SIZE) - 1


class Op(ABC):
    """Base class for immutable regular-expression IR operations."""

    @property
    @abstractmethod
    def operands(self) -> tuple[Op, ...]:
        """Return the operation's ordered child operations."""

    @abstractmethod
    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        """Clone the operation with a new ordered operand list."""

    def verify(self) -> None:
        if any(not isinstance(operand, Op) for operand in self.operands):
            raise TypeError("every REIR operand must be an Op")


class PureOp(Op, ABC):
    """Marker base for operations with pure regular-language semantics."""


class LeafOp(PureOp, ABC):
    """Base class for operations without operands."""

    @property
    def operands(self) -> tuple[Op, ...]:
        return ()

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        if operands:
            raise ValueError(f"{type(self).__name__} does not accept operands")
        return self


@dataclass(frozen=True, slots=True)
class Never(LeafOp):
    """The empty language."""


@dataclass(frozen=True, slots=True)
class Epsilon(LeafOp):
    """The language containing only the empty byte string."""


@dataclass(frozen=True, slots=True, init=False)
class CharSet(LeafOp):
    """A non-empty subset of the byte alphabet stored as a canonical bitset."""

    bits: int

    def __init__(self, symbols: Iterable[int]) -> None:
        bits = 0
        for symbol in symbols:
            if not 0 <= symbol < BYTE_ALPHABET_SIZE:
                raise ValueError(
                    f"character-set symbol is outside the byte alphabet: {symbol}"
                )
            bits |= 1 << symbol
        if not bits:
            raise ValueError("a CharSet operation must not be empty")
        object.__setattr__(self, "bits", bits)

    @classmethod
    def from_bits(cls, bits: int) -> CharSet:
        if bits <= 0 or bits & ~BYTE_ALPHABET_MASK:
            raise ValueError("a CharSet bitset must be a non-empty byte subset")
        result = object.__new__(cls)
        object.__setattr__(result, "bits", bits)
        return result

    @property
    def symbols(self) -> frozenset[int]:
        return frozenset(
            symbol for symbol in range(BYTE_ALPHABET_SIZE) if self.bits & (1 << symbol)
        )

    @property
    def intervals(self) -> tuple[tuple[int, int], ...]:
        intervals: list[tuple[int, int]] = []
        start: int | None = None
        previous = -1
        for symbol in range(BYTE_ALPHABET_SIZE):
            if not self.bits & (1 << symbol):
                continue
            if start is None:
                start = symbol
            elif symbol != previous + 1:
                intervals.append((start, previous))
                start = symbol
            previous = symbol
        assert start is not None
        intervals.append((start, previous))
        return tuple(intervals)

    def complement(self) -> Op:
        complement = BYTE_ALPHABET_MASK ^ self.bits
        return CharSet.from_bits(complement) if complement else NEVER


@dataclass(frozen=True, slots=True)
class Literal(LeafOp):
    value: bytes

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("a regex literal must not be empty")


@dataclass(frozen=True, slots=True)
class Concat(PureOp):
    parts: tuple[Op, ...]

    def __post_init__(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("a regex concatenation requires at least two parts")
        if any(not isinstance(part, PureOp) for part in self.parts):
            raise TypeError("pure Concat operands must be PureOp instances")

    @property
    def operands(self) -> tuple[Op, ...]:
        return self.parts

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        return Concat(operands)


@dataclass(frozen=True, slots=True)
class Alternate(PureOp):
    alternatives: tuple[Op, ...]

    def __post_init__(self) -> None:
        if len(self.alternatives) < 2:
            raise ValueError("a regex alternation requires at least two branches")
        if any(not isinstance(branch, PureOp) for branch in self.alternatives):
            raise TypeError("pure Alternate operands must be PureOp instances")

    @property
    def operands(self) -> tuple[Op, ...]:
        return self.alternatives

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        return Alternate(operands)


@dataclass(frozen=True, slots=True)
class Repeat(PureOp):
    body: Op
    min: int
    max: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.body, PureOp):
            raise TypeError("a pure Repeat body must be a PureOp instance")
        if self.min < 0:
            raise ValueError("a regex repetition minimum must be non-negative")
        if self.max is not None and self.max < self.min:
            raise ValueError("a regex repetition maximum must not be below its minimum")

    @property
    def operands(self) -> tuple[Op, ...]:
        return (self.body,)

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        if len(operands) != 1:
            raise ValueError("a regex repetition requires exactly one body")
        return Repeat(operands[0], self.min, self.max)


type PureRegex = Never | Epsilon | CharSet | Literal | Concat | Alternate | Repeat

NEVER = Never()
EPSILON = Epsilon()


def structural_key(op: Op) -> tuple[Any, ...]:
    """Return a deterministic total-order key for core pure regex operations."""
    match op:
        case Never():
            return (0,)
        case Epsilon():
            return (1,)
        case CharSet():
            return (2, op.bits)
        case Literal(value):
            return (3, value)
        case Repeat(body, minimum, maximum):
            maximum_key = (1, 0) if maximum is None else (0, maximum)
            return (4, structural_key(body), minimum, maximum_key)
        case Concat(parts):
            return (5, tuple(structural_key(part) for part in parts))
        case Alternate(alternatives):
            return (6, tuple(structural_key(item) for item in alternatives))
        case _ if isinstance(op, PureOp):
            return (
                7,
                type(op).__module__,
                type(op).__qualname__,
                tuple(structural_key(operand) for operand in op.operands),
            )
        case _:
            raise TypeError(f"{type(op).__name__} does not have pure regex semantics")


__all__ = [
    "BYTE_ALPHABET_MASK",
    "BYTE_ALPHABET_SIZE",
    "EPSILON",
    "NEVER",
    "Alternate",
    "CharSet",
    "Concat",
    "Epsilon",
    "LeafOp",
    "Literal",
    "Never",
    "Op",
    "PureOp",
    "PureRegex",
    "Repeat",
    "structural_key",
]
