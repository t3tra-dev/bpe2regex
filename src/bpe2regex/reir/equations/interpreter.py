"""Exact reference interpreter for length-guarded regular equations."""

from collections import defaultdict
from collections.abc import Callable
from functools import cache

from .analysis import verify_length_guarded
from .ir import (
    EquationAlternate,
    EquationConcat,
    EquationDifference,
    EquationEpsilon,
    EquationExpr,
    EquationLiteral,
    EquationNever,
    EquationRef,
    EquationTokenSelectorRef,
    EquationUniversal,
    RegularEquationIR,
)


class RegularEquationInterpreter:
    """Evaluate equation languages without choosing a fixed-point iteration.

    Guarded references always advance the byte position, so recursive calls are
    well-founded on the remaining input length.  Returning all possible end
    positions also makes concatenation and language difference exact.
    """

    def __init__(self, ir: RegularEquationIR) -> None:
        verify_length_guarded(ir)
        self.ir = ir
        by_first: dict[int, list[int]] = defaultdict(list)
        for token in ir.active_tokens:
            value = ir.tokens[token]
            if value is None or not value:
                raise AssertionError("verified active equation tokens are non-empty")
            by_first[value[0]].append(token)
        self._tokens_by_first = {
            byte: tuple(sorted(ranks)) for byte, ranks in by_first.items()
        }

    def _evaluator(
        self,
        value: bytes,
    ) -> tuple[
        Callable[[EquationExpr, int], frozenset[int]],
        Callable[[int, int], frozenset[int]],
    ]:
        length = len(value)

        @cache
        def state_positions(state: int, position: int) -> frozenset[int]:
            return positions(self.ir.equation(state).body, position)

        @cache
        def positions(expression: EquationExpr, position: int) -> frozenset[int]:
            match expression:
                case EquationNever():
                    return frozenset()
                case EquationEpsilon():
                    return frozenset((position,))
                case EquationUniversal():
                    ends: list[int] = [position]
                    for end in range(position, length):
                        if value[end] not in self.ir.alphabet:
                            break
                        ends.append(end + 1)
                    return frozenset(ends)
                case EquationLiteral(literal):
                    end = position + len(literal)
                    return (
                        frozenset((end,))
                        if end <= length and value.startswith(literal, position)
                        else frozenset()
                    )
                case EquationRef(variable):
                    return state_positions(variable, position)
                case EquationTokenSelectorRef(selector):
                    if position >= length:
                        return frozenset()
                    selector_root = self.ir.selector_dag.selectors[selector]
                    result: set[int] = set()
                    for token in self._tokens_by_first.get(value[position], ()):
                        if not selector_root.all_tokens and not self.ir.selector_dag.contains(
                            selector, token
                        ):
                            continue
                        token_bytes = self.ir.tokens[token]
                        if token_bytes is None:
                            raise AssertionError("a selected token must be active")
                        end = position + len(token_bytes)
                        if end <= length and value.startswith(token_bytes, position):
                            result.update(
                                state_positions(self.ir.token_target(token), end)
                            )
                    return frozenset(result)
                case EquationConcat(parts):
                    current = frozenset((position,))
                    for part in parts:
                        current = frozenset(
                            end
                            for start in current
                            for end in positions(part, start)
                        )
                        if not current:
                            break
                    return current
                case EquationAlternate(alternatives):
                    return frozenset(
                        end
                        for branch in alternatives
                        for end in positions(branch, position)
                    )
                case EquationDifference(left, right):
                    return positions(left, position) - positions(right, position)
            raise TypeError(
                f"unsupported equation expression: {type(expression).__name__}"
            )

        return positions, state_positions

    def matches(self, expression: EquationExpr, value: bytes) -> bool:
        subject = bytes(value)
        positions, _ = self._evaluator(subject)
        return len(subject) in positions(expression, 0)

    def matches_entry(self, value: bytes) -> bool:
        subject = bytes(value)
        positions, _ = self._evaluator(subject)
        return len(subject) in positions(self.ir.entry, 0)

    def matches_state(self, state: int, value: bytes) -> bool:
        subject = bytes(value)
        _, state_positions = self._evaluator(subject)
        return len(subject) in state_positions(state, 0)

    def matching_first_tokens(self, value: bytes) -> tuple[int, ...]:
        subject = bytes(value)
        if not subject:
            return ()
        _, state_positions = self._evaluator(subject)
        denied = self.ir.denial_selector(self.ir.start)
        selected: list[int] = []
        for token in self._tokens_by_first.get(subject[0], ()):
            if self.ir.selector_dag.contains(denied, token):
                continue
            token_bytes = self.ir.tokens[token]
            if token_bytes is None:
                raise AssertionError("a first-token candidate must be active")
            end = len(token_bytes)
            if end <= len(subject) and subject.startswith(token_bytes):
                target_ends = state_positions(self.ir.token_target(token), end)
                if len(subject) in target_ends:
                    selected.append(token)
        return tuple(selected)

    def first_token(self, value: bytes) -> int | None:
        subject = bytes(value)
        if not subject:
            return None
        selected = self.matching_first_tokens(subject)
        if len(selected) != 1:
            raise ValueError(
                "canonical equation entry must select exactly one first token; "
                f"observed {selected!r}"
            )
        return selected[0]

    def tokenize(self, value: bytes) -> tuple[int, ...]:
        subject = bytes(value)
        position = 0
        result: list[int] = []
        while position < len(subject):
            token = self.first_token(subject[position:])
            if token is None:
                raise AssertionError("a non-empty suffix must select a token")
            token_bytes = self.ir.tokens[token]
            if token_bytes is None or not token_bytes:
                raise AssertionError("a selected token must make progress")
            position += len(token_bytes)
            result.append(token)
        return tuple(result)


__all__ = ["RegularEquationInterpreter"]
