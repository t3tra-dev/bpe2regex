from dataclasses import dataclass

from .analysis import AnalysisManager, RegexPropertiesAnalysis
from .builder import DEFAULT_BUILDER, RegexBuilder
from .ops import (
    BYTE_ALPHABET_SIZE,
    EPSILON,
    NEVER,
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


@dataclass(frozen=True, slots=True)
class DerivativeGroup:
    """Input bytes whose canonical derivative IR is structurally equal."""

    symbols: CharSet
    residual: Op

    def __post_init__(self) -> None:
        if not isinstance(self.symbols, CharSet):
            raise TypeError("a derivative group requires a non-empty CharSet")
        if not isinstance(self.residual, PureOp):
            raise TypeError("a derivative residual must have pure regex semantics")


class DerivativeEngine:
    """Memoized Brzozowski derivatives over the seven pure REIR operations."""

    def __init__(
        self,
        *,
        builder: RegexBuilder = DEFAULT_BUILDER,
        analyses: AnalysisManager | None = None,
    ) -> None:
        self.builder = builder
        self.analyses = AnalysisManager() if analyses is None else analyses
        self._cache: dict[tuple[int, int], tuple[Op, Op]] = {}
        self._active: set[tuple[int, int]] = set()

    @property
    def cached_derivative_count(self) -> int:
        return len(self._cache)

    def cached_symbols(self, root: Op) -> frozenset[int]:
        """Return the byte derivatives memoized specifically for ``root``."""
        identity = id(root)
        return frozenset(
            symbol
            for (cached_identity, symbol), (cached_root, _) in self._cache.items()
            if cached_identity == identity and cached_root is root
        )

    def derive(self, root: Op, symbol: int) -> Op:
        if not isinstance(root, PureOp):
            raise TypeError("derivatives require pure regex semantics")
        if not isinstance(symbol, int):
            raise TypeError("a derivative symbol must be an integer byte")
        if not 0 <= symbol < BYTE_ALPHABET_SIZE:
            raise ValueError("a derivative symbol must belong to the byte alphabet")

        key = id(root), symbol
        known = self._cache.get(key)
        if known is not None and known[0] is root:
            return known[1]
        if key in self._active:
            raise ValueError("REIR derivatives require an acyclic operation graph")
        self._active.add(key)
        try:
            properties = self.analyses.get(RegexPropertiesAnalysis, root)
            result = (
                self._derive_uncached(root, symbol)
                if symbol in properties.first_symbols
                else NEVER
            )
            if not isinstance(result, PureOp):
                raise TypeError("a derivative produced non-pure REIR")
            result.verify()
            self._cache[key] = root, result
            return result
        finally:
            self._active.remove(key)

    def _derive_uncached(self, root: Op, symbol: int) -> Op:
        match root:
            case Never() | Epsilon():
                return NEVER
            case CharSet():
                return EPSILON if root.bits & (1 << symbol) else NEVER
            case Literal(value):
                if value[0] != symbol:
                    return NEVER
                return self.builder.literal(value[1:])
            case Alternate(branches):
                return self.builder.alternate(
                    *(self.derive(branch, symbol) for branch in branches)
                )
            case Concat(parts):
                derivatives: list[Op] = []
                for index, part in enumerate(parts):
                    derivatives.append(
                        self.builder.concat(
                            self.derive(part, symbol),
                            *parts[index + 1 :],
                        )
                    )
                    properties = self.analyses.get(RegexPropertiesAnalysis, part)
                    if not properties.nullable:
                        break
                return self.builder.alternate(*derivatives)
            case Repeat(body, minimum, maximum):
                if maximum == 0:
                    return NEVER
                body_properties = self.analyses.get(RegexPropertiesAnalysis, body)
                tail_minimum = 0 if body_properties.nullable else max(0, minimum - 1)
                tail_maximum = None if maximum is None else maximum - 1
                return self.builder.concat(
                    self.derive(body, symbol),
                    self.builder.repeat(body, tail_minimum, tail_maximum),
                )
            case _:
                raise TypeError(
                    f"DerivativeEngine has no rule for {type(root).__name__}"
                )

    def group(self, root: Op) -> tuple[DerivativeGroup, ...]:
        """Evaluate only ``First(root)`` and group bytes by equal residuals."""
        if not isinstance(root, PureOp):
            raise TypeError("derivative grouping requires pure regex semantics")
        properties = self.analyses.get(RegexPropertiesAnalysis, root)
        residual_bits: dict[Op, int] = {}
        for symbol in sorted(properties.first_symbols):
            residual = self.derive(root, symbol)
            residual_bits[residual] = residual_bits.get(residual, 0) | (1 << symbol)
        return tuple(
            DerivativeGroup(CharSet.from_bits(bits), residual)
            for residual, bits in residual_bits.items()
        )


def derivative(
    root: Op,
    symbol: int,
    *,
    builder: RegexBuilder = DEFAULT_BUILDER,
    analyses: AnalysisManager | None = None,
) -> Op:
    return DerivativeEngine(builder=builder, analyses=analyses).derive(root, symbol)


def group_derivatives(
    root: Op,
    *,
    builder: RegexBuilder = DEFAULT_BUILDER,
    analyses: AnalysisManager | None = None,
) -> tuple[DerivativeGroup, ...]:
    return DerivativeEngine(builder=builder, analyses=analyses).group(root)


__all__ = [
    "DerivativeEngine",
    "DerivativeGroup",
    "derivative",
    "group_derivatives",
]
