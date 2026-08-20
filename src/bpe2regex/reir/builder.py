from collections.abc import Iterable

from .analysis import AnalysisManager
from .ops import (
    EPSILON,
    NEVER,
    Alternate,
    CharSet,
    Concat,
    Literal,
    Op,
    PureOp,
    Repeat,
)
from .rewrite import (
    CANONICALIZATION_PATTERNS,
    GreedyRewriteDriver,
    PatternApplicator,
    RewritePattern,
)


class RegexBuilder:
    """Construct canonical byte-regex REIR with extensible rewrite patterns."""

    def __init__(
        self,
        patterns: Iterable[RewritePattern] = CANONICALIZATION_PATTERNS,
    ) -> None:
        self.patterns = tuple(patterns)
        self._applicator = PatternApplicator(self.patterns)

    def _check_operands(self, operands: Iterable[Op]) -> None:
        if any(not isinstance(operand, PureOp) for operand in operands):
            raise TypeError("RegexBuilder only accepts pure regex operations")

    def charset(self, symbols: Iterable[int]) -> Op:
        canonical = frozenset(symbols)
        if not canonical:
            return NEVER
        return self._applicator.rewrite_root(CharSet(canonical))

    def literal(self, value: bytes | bytearray | memoryview) -> Op:
        content = bytes(value)
        return self._applicator.rewrite_root(Literal(content) if content else EPSILON)

    def concat(self, *parts: Op) -> Op:
        self._check_operands(parts)
        if not parts:
            return EPSILON
        if len(parts) == 1:
            return self._applicator.rewrite_root(parts[0])
        return self._applicator.rewrite_root(Concat(tuple(parts)))

    def alternate(self, *alternatives: Op) -> Op:
        self._check_operands(alternatives)
        if not alternatives:
            return NEVER
        if len(alternatives) == 1:
            return self._applicator.rewrite_root(alternatives[0])
        return self._applicator.rewrite_root(Alternate(tuple(alternatives)))

    def repeat(self, body: Op, minimum: int, maximum: int | None) -> Op:
        self._check_operands((body,))
        return self._applicator.rewrite_root(Repeat(body, minimum, maximum))

    def normalize(
        self,
        root: Op,
        analyses: AnalysisManager | None = None,
    ) -> Op:
        self._check_operands((root,))
        return GreedyRewriteDriver(self.patterns).rewrite(root, analyses)


DEFAULT_BUILDER = RegexBuilder()


def charset(symbols: Iterable[int]) -> Op:
    return DEFAULT_BUILDER.charset(symbols)


def literal(value: bytes | bytearray | memoryview) -> Op:
    return DEFAULT_BUILDER.literal(value)


def concat(*parts: Op) -> Op:
    return DEFAULT_BUILDER.concat(*parts)


def alternate(*alternatives: Op) -> Op:
    return DEFAULT_BUILDER.alternate(*alternatives)


def repeat(body: Op, minimum: int, maximum: int | None) -> Op:
    return DEFAULT_BUILDER.repeat(body, minimum, maximum)


__all__ = [
    "DEFAULT_BUILDER",
    "RegexBuilder",
    "alternate",
    "charset",
    "concat",
    "literal",
    "repeat",
]
