from collections.abc import Iterable

from ..builder import DEFAULT_BUILDER, RegexBuilder
from ..ops import NEVER, Never, Op, PureOp, structural_key
from .ops import UNIVERSAL, Complement, Difference, Intersect, Universal


class BooleanRegexBuilder:
    """Build a canonical transient Boolean regex dialect over core REIR."""

    def __init__(self, *, core_builder: RegexBuilder = DEFAULT_BUILDER) -> None:
        self.core_builder = core_builder

    @staticmethod
    def _check_pure(expressions: Iterable[Op]) -> None:
        if any(not isinstance(expression, PureOp) for expression in expressions):
            raise TypeError("Boolean regex operations require pure regex semantics")

    def complement(self, body: Op) -> Op:
        self._check_pure((body,))
        if isinstance(body, Never):
            return UNIVERSAL
        if isinstance(body, Universal):
            return NEVER
        if isinstance(body, Complement):
            return body.body
        return Complement(body)

    def intersect(self, *expressions: Op) -> Op:
        self._check_pure(expressions)
        flattened: list[Op] = []
        pending = list(reversed(expressions))
        while pending:
            expression = pending.pop()
            if isinstance(expression, Never):
                return NEVER
            if isinstance(expression, Universal):
                continue
            if isinstance(expression, Intersect):
                pending.extend(reversed(expression.expressions))
            else:
                flattened.append(expression)

        unique = list(dict.fromkeys(flattened))
        expression_set = frozenset(unique)
        if any(
            isinstance(expression, Complement) and expression.body in expression_set
            for expression in unique
        ):
            return NEVER
        unique.sort(key=structural_key)
        if not unique:
            return UNIVERSAL
        if len(unique) == 1:
            return unique[0]
        return Intersect(tuple(unique))

    def difference(self, left: Op, right: Op) -> Op:
        self._check_pure((left, right))
        if isinstance(left, Never) or left == right or isinstance(right, Universal):
            return NEVER
        if isinstance(right, Never):
            return left
        if isinstance(left, Universal):
            return self.complement(right)
        if isinstance(right, Complement):
            return self.intersect(left, right.body)
        return Difference(left, right)

    def concat(self, *expressions: Op) -> Op:
        self._check_pure(expressions)
        return self.core_builder.concat(*expressions)

    def alternate(self, *expressions: Op) -> Op:
        self._check_pure(expressions)
        if any(isinstance(expression, Universal) for expression in expressions):
            return UNIVERSAL
        return self.core_builder.alternate(*expressions)

    def repeat(self, body: Op, minimum: int, maximum: int | None) -> Op:
        self._check_pure((body,))
        if isinstance(body, Universal):
            if maximum == 0:
                return self.core_builder.repeat(body, minimum, maximum)
            return UNIVERSAL
        return self.core_builder.repeat(body, minimum, maximum)


BOOLEAN_BUILDER = BooleanRegexBuilder()


def complement(body: Op) -> Op:
    return BOOLEAN_BUILDER.complement(body)


def intersect(*expressions: Op) -> Op:
    return BOOLEAN_BUILDER.intersect(*expressions)


def difference(left: Op, right: Op) -> Op:
    return BOOLEAN_BUILDER.difference(left, right)


__all__ = [
    "BOOLEAN_BUILDER",
    "BooleanRegexBuilder",
    "complement",
    "difference",
    "intersect",
]
