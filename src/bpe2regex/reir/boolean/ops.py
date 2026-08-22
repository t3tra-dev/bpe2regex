from dataclasses import dataclass

from ..ops import Op, PureOp


@dataclass(frozen=True, slots=True)
class Universal(PureOp):
    """The universal byte-string language, ``Sigma*``."""

    @property
    def operands(self) -> tuple[Op, ...]:
        return ()

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        if operands:
            raise ValueError("Universal does not accept operands")
        return self


@dataclass(frozen=True, slots=True)
class Intersect(PureOp):
    """An n-ary regular-language intersection."""

    expressions: tuple[Op, ...]

    def __post_init__(self) -> None:
        if len(self.expressions) < 2:
            raise ValueError("an intersection requires at least two operands")
        if any(not isinstance(expression, PureOp) for expression in self.expressions):
            raise TypeError("intersection operands must have pure regex semantics")

    @property
    def operands(self) -> tuple[Op, ...]:
        return self.expressions

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        return Intersect(operands)


@dataclass(frozen=True, slots=True)
class Complement(PureOp):
    """The complement of a byte-string language relative to ``Sigma*``."""

    body: Op

    def __post_init__(self) -> None:
        if not isinstance(self.body, PureOp):
            raise TypeError("a complement body must have pure regex semantics")

    @property
    def operands(self) -> tuple[Op, ...]:
        return (self.body,)

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        if len(operands) != 1:
            raise ValueError("a complement requires exactly one operand")
        return Complement(operands[0])


@dataclass(frozen=True, slots=True)
class Difference(PureOp):
    """Regular-language difference, ``left \\ right``."""

    left: Op
    right: Op

    def __post_init__(self) -> None:
        if not isinstance(self.left, PureOp) or not isinstance(self.right, PureOp):
            raise TypeError("difference operands must have pure regex semantics")

    @property
    def operands(self) -> tuple[Op, ...]:
        return self.left, self.right

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        if len(operands) != 2:
            raise ValueError("a difference requires exactly two operands")
        return Difference(*operands)


UNIVERSAL = Universal()

type BooleanOp = Universal | Intersect | Complement | Difference


__all__ = [
    "UNIVERSAL",
    "BooleanOp",
    "Complement",
    "Difference",
    "Intersect",
    "Universal",
]
