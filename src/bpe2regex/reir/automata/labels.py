from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ..ops import BYTE_ALPHABET_SIZE, NEVER, CharSet, Op


@dataclass(frozen=True, slots=True)
class SymbolSet:
    """A canonical finite-alphabet symbol set stored as a bitset."""

    alphabet_size: int
    bits: int

    def __post_init__(self) -> None:
        if self.alphabet_size <= 0:
            raise ValueError("a symbol alphabet must contain at least one symbol")
        if self.bits < 0 or self.bits & ~self.mask:
            raise ValueError("symbol-set bits are outside the declared alphabet")

    @classmethod
    def empty(cls, alphabet_size: int) -> SymbolSet:
        return cls(alphabet_size, 0)

    @classmethod
    def full(cls, alphabet_size: int) -> SymbolSet:
        if alphabet_size <= 0:
            raise ValueError("a symbol alphabet must contain at least one symbol")
        return cls(alphabet_size, (1 << alphabet_size) - 1)

    @classmethod
    def singleton(cls, alphabet_size: int, symbol: int) -> SymbolSet:
        return cls.from_symbols(alphabet_size, (symbol,))

    @classmethod
    def from_symbols(
        cls,
        alphabet_size: int,
        symbols: Iterable[int],
    ) -> SymbolSet:
        if alphabet_size <= 0:
            raise ValueError("a symbol alphabet must contain at least one symbol")
        bits = 0
        for symbol in symbols:
            if not isinstance(symbol, int):
                raise TypeError("an automaton symbol must be an integer")
            if not 0 <= symbol < alphabet_size:
                raise ValueError("a symbol is outside the declared alphabet")
            bits |= 1 << symbol
        return cls(alphabet_size, bits)

    @classmethod
    def from_charset(cls, symbols: CharSet) -> SymbolSet:
        return cls(BYTE_ALPHABET_SIZE, symbols.bits)

    @property
    def mask(self) -> int:
        return (1 << self.alphabet_size) - 1

    @property
    def symbols(self) -> frozenset[int]:
        return frozenset(self)

    @property
    def first_symbol(self) -> int | None:
        if not self.bits:
            return None
        return (self.bits & -self.bits).bit_length() - 1

    @property
    def intervals(self) -> tuple[tuple[int, int], ...]:
        intervals: list[tuple[int, int]] = []
        start: int | None = None
        previous = -1
        for symbol in self:
            if start is None:
                start = symbol
            elif symbol != previous + 1:
                intervals.append((start, previous))
                start = symbol
            previous = symbol
        if start is not None:
            intervals.append((start, previous))
        return tuple(intervals)

    def _check_compatible(self, other: SymbolSet) -> None:
        if self.alphabet_size != other.alphabet_size:
            raise ValueError("symbol sets belong to different alphabets")

    def union(self, other: SymbolSet) -> SymbolSet:
        self._check_compatible(other)
        return SymbolSet(self.alphabet_size, self.bits | other.bits)

    def intersection(self, other: SymbolSet) -> SymbolSet:
        self._check_compatible(other)
        return SymbolSet(self.alphabet_size, self.bits & other.bits)

    def difference(self, other: SymbolSet) -> SymbolSet:
        self._check_compatible(other)
        return SymbolSet(self.alphabet_size, self.bits & ~other.bits)

    def symmetric_difference(self, other: SymbolSet) -> SymbolSet:
        self._check_compatible(other)
        return SymbolSet(self.alphabet_size, self.bits ^ other.bits)

    def complement(self) -> SymbolSet:
        return SymbolSet(self.alphabet_size, self.mask ^ self.bits)

    def isdisjoint(self, other: SymbolSet) -> bool:
        self._check_compatible(other)
        return not self.bits & other.bits

    def issubset(self, other: SymbolSet) -> bool:
        self._check_compatible(other)
        return not self.bits & ~other.bits

    def to_reir(self) -> Op:
        if self.alphabet_size != BYTE_ALPHABET_SIZE:
            raise ValueError("only the byte alphabet can lower to a REIR CharSet")
        return CharSet.from_bits(self.bits) if self.bits else NEVER

    def __or__(self, other: SymbolSet) -> SymbolSet:
        return self.union(other)

    def __and__(self, other: SymbolSet) -> SymbolSet:
        return self.intersection(other)

    def __sub__(self, other: SymbolSet) -> SymbolSet:
        return self.difference(other)

    def __xor__(self, other: SymbolSet) -> SymbolSet:
        return self.symmetric_difference(other)

    def __contains__(self, symbol: object) -> bool:
        return (
            isinstance(symbol, int)
            and 0 <= symbol < self.alphabet_size
            and bool(self.bits & (1 << symbol))
        )

    def __iter__(self) -> Iterator[int]:
        pending = self.bits
        while pending:
            lowest = pending & -pending
            yield lowest.bit_length() - 1
            pending ^= lowest

    def __len__(self) -> int:
        return self.bits.bit_count()

    def __bool__(self) -> bool:
        return bool(self.bits)


__all__ = ["SymbolSet"]
