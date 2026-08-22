from ..ops import (
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
from .analysis import BooleanRegexPropertiesAnalysis
from .builder import BOOLEAN_BUILDER, BooleanRegexBuilder
from .ops import UNIVERSAL, Complement, Difference, Intersect, Universal


class BooleanDerivativeEngine:
    """Memoized Brzozowski derivatives for core and Boolean pure REIR."""

    def __init__(
        self,
        *,
        builder: BooleanRegexBuilder = BOOLEAN_BUILDER,
        analyses: BooleanRegexPropertiesAnalysis | None = None,
    ) -> None:
        self.builder = builder
        self.analyses = (
            BooleanRegexPropertiesAnalysis() if analyses is None else analyses
        )
        self._cache: dict[tuple[Op, int], PureOp] = {}
        self._active: set[tuple[int, int]] = set()

    @property
    def cached_derivative_count(self) -> int:
        return len(self._cache)

    def derive(self, root: Op, symbol: int) -> PureOp:
        if not isinstance(root, PureOp):
            raise TypeError("Boolean derivatives require pure regex semantics")
        if not isinstance(symbol, int):
            raise TypeError("a derivative symbol must be an integer byte")
        if not 0 <= symbol < BYTE_ALPHABET_SIZE:
            raise ValueError("a derivative symbol must belong to the byte alphabet")
        key = root, symbol
        known = self._cache.get(key)
        if known is not None:
            return known
        active_key = id(root), symbol
        if active_key in self._active:
            raise ValueError("Boolean REIR derivatives require an acyclic graph")
        self._active.add(active_key)
        try:
            result = (
                self._derive_uncached(root, symbol)
                if symbol in self.analyses.get(root).first_symbols
                else NEVER
            )
            if not isinstance(result, PureOp):
                raise TypeError("a Boolean derivative produced non-pure REIR")
            result.verify()
            self._cache[key] = result
            return result
        finally:
            self._active.remove(active_key)

    def _derive_uncached(self, root: Op, symbol: int) -> Op:
        match root:
            case Never() | Epsilon():
                return NEVER
            case Universal():
                return UNIVERSAL
            case CharSet():
                return EPSILON if root.bits & (1 << symbol) else NEVER
            case Literal(value):
                if value[0] != symbol:
                    return NEVER
                return self.builder.core_builder.literal(value[1:])
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
                    if not self.analyses.get(part).nullable:
                        break
                return self.builder.alternate(*derivatives)
            case Repeat(body, minimum, maximum):
                if maximum == 0:
                    return NEVER
                body_nullable = self.analyses.get(body).nullable
                tail_minimum = 0 if body_nullable else max(0, minimum - 1)
                tail_maximum = None if maximum is None else maximum - 1
                return self.builder.concat(
                    self.derive(body, symbol),
                    self.builder.repeat(body, tail_minimum, tail_maximum),
                )
            case Intersect(expressions):
                return self.builder.intersect(
                    *(self.derive(expression, symbol) for expression in expressions)
                )
            case Complement(body):
                return self.builder.complement(self.derive(body, symbol))
            case Difference(left, right):
                return self.builder.difference(
                    self.derive(left, symbol),
                    self.derive(right, symbol),
                )
            case _:
                raise TypeError(
                    f"BooleanDerivativeEngine has no rule for {type(root).__name__}"
                )


def boolean_derivative(root: Op, symbol: int) -> Op:
    return BooleanDerivativeEngine().derive(root, symbol)


__all__ = ["BooleanDerivativeEngine", "boolean_derivative"]
