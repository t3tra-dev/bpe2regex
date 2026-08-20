from collections.abc import Iterable
from dataclasses import dataclass

from .builder import DEFAULT_BUILDER
from .ops import (
    EPSILON,
    NEVER,
    Concat,
    Epsilon,
    Literal,
    Never,
    Op,
    PureOp,
    PureRegex,
)


@dataclass(frozen=True, slots=True)
class Tag(Op):
    """A zero-width transducer output outside pure regular-language semantics."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("a regex tag must be non-negative")

    @property
    def operands(self) -> tuple[Op, ...]:
        return ()

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        if operands:
            raise ValueError("Tag does not accept operands")
        return self


@dataclass(frozen=True, slots=True)
class TaggedConcat(Op):
    parts: tuple[Op, ...]

    def __post_init__(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("a tagged concatenation requires at least two parts")

    @property
    def operands(self) -> tuple[Op, ...]:
        return self.parts

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        return TaggedConcat(operands)


@dataclass(frozen=True, slots=True)
class TaggedAlternate(Op):
    alternatives: tuple[Op, ...]

    def __post_init__(self) -> None:
        if len(self.alternatives) < 2:
            raise ValueError("a tagged alternation requires at least two branches")

    @property
    def operands(self) -> tuple[Op, ...]:
        return self.alternatives

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        return TaggedAlternate(operands)


type TaggedRegex = PureRegex | Tag | TaggedConcat | TaggedAlternate


class TaggedRegexBuilder:
    """Build ordered output regexes while delegating pure subgraphs to REIR."""

    def charset(self, symbols: Iterable[int]) -> Op:
        return DEFAULT_BUILDER.charset(symbols)

    def literal(self, value: bytes | bytearray | memoryview) -> Op:
        return DEFAULT_BUILDER.literal(value)

    def repeat(self, body: Op, minimum: int, maximum: int | None) -> Op:
        if not isinstance(body, PureOp):
            raise TypeError("Repeat is not defined for tagged output semantics")
        return DEFAULT_BUILDER.repeat(body, minimum, maximum)

    def tag(self, value: int) -> Tag:
        return Tag(value)

    def concat(self, *parts: Op) -> Op:
        if all(isinstance(part, PureOp) for part in parts):
            return DEFAULT_BUILDER.concat(*parts)

        flattened: list[Op] = []
        pending_literal = bytearray()

        def flush_literal() -> None:
            if pending_literal:
                flattened.append(Literal(bytes(pending_literal)))
                pending_literal.clear()

        pending = list(reversed(parts))
        while pending:
            part = pending.pop()
            if isinstance(part, Never):
                return NEVER
            if isinstance(part, Epsilon):
                continue
            if isinstance(part, (Concat, TaggedConcat)):
                pending.extend(reversed(part.operands))
            elif isinstance(part, Literal):
                pending_literal.extend(part.value)
            else:
                flush_literal()
                flattened.append(part)
        flush_literal()
        if not flattened:
            return EPSILON
        if len(flattened) == 1:
            return flattened[0]
        if all(isinstance(part, PureOp) for part in flattened):
            return DEFAULT_BUILDER.concat(*flattened)
        return TaggedConcat(tuple(flattened))

    def alternate(self, *alternatives: Op) -> Op:
        if all(isinstance(branch, PureOp) for branch in alternatives):
            return DEFAULT_BUILDER.alternate(*alternatives)

        flattened: list[Op] = []
        pending = list(reversed(alternatives))
        while pending:
            alternative = pending.pop()
            if isinstance(alternative, Never):
                continue
            if isinstance(alternative, TaggedAlternate):
                pending.extend(reversed(alternative.alternatives))
            else:
                flattened.append(alternative)
        if not flattened:
            return NEVER
        if len(flattened) == 1:
            return flattened[0]
        if all(isinstance(branch, PureOp) for branch in flattened):
            return DEFAULT_BUILDER.alternate(*flattened)
        return TaggedAlternate(tuple(flattened))


TAGGED_BUILDER = TaggedRegexBuilder()


def tagged(value: int) -> Tag:
    return TAGGED_BUILDER.tag(value)


__all__ = [
    "TAGGED_BUILDER",
    "Tag",
    "TaggedAlternate",
    "TaggedConcat",
    "TaggedRegex",
    "TaggedRegexBuilder",
    "tagged",
]
