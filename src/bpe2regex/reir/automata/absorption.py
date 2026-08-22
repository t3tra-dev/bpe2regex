from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from .algorithms import combined_alphabet_partition
from .ir import DFA

type ProductState = tuple[int | None, int | None]


def acceptance_inclusion_counterexample(
    subset: DFA[bool],
    superset: DFA[bool],
) -> tuple[int, ...] | None:
    """Return the shortest lexicographic word in ``subset - superset``."""
    symbol_classes = combined_alphabet_partition(subset, superset)
    symbols = tuple(symbol_class.first_symbol for symbol_class in symbol_classes)
    if any(symbol is None for symbol in symbols):
        raise AssertionError("an alphabet partition must not contain empty classes")

    start: ProductState = subset.start, superset.start
    pending: deque[ProductState] = deque((start,))
    parents: dict[ProductState, tuple[ProductState, int] | None] = {start: None}

    def accepts(automaton: DFA[bool], state: int | None) -> bool:
        return state is not None and automaton.outputs[state] is not None

    def word(state: ProductState) -> tuple[int, ...]:
        result: list[int] = []
        current = state
        parent = parents[current]
        while parent is not None:
            previous, symbol = parent
            result.append(symbol)
            current = previous
            parent = parents[current]
        result.reverse()
        return tuple(result)

    while pending:
        state = pending.popleft()
        if accepts(subset, state[0]) and not accepts(superset, state[1]):
            return word(state)
        for symbol in symbols:
            assert symbol is not None
            successor: ProductState = (
                None if state[0] is None else subset.transition(state[0], symbol),
                None if state[1] is None else superset.transition(state[1], symbol),
            )
            if successor not in parents:
                parents[successor] = state, symbol
                pending.append(successor)
    return None


def acceptance_included(subset: DFA[bool], superset: DFA[bool]) -> bool:
    return acceptance_inclusion_counterexample(subset, superset) is None


@dataclass(frozen=True, slots=True)
class SemanticAbsorptionResult:
    """A pure union after language-included alternatives have been absorbed."""

    alternatives: tuple[DFA[bool], ...]
    kept_indices: tuple[int, ...]
    absorbed_by: tuple[int | None, ...]
    comparison_count: int


class AutomatonSemanticAbsorber:
    """Expensive pairwise language-inclusion simplification for pure DFA unions."""

    def run(self, alternatives: Iterable[DFA[bool]]) -> SemanticAbsorptionResult:
        candidates = tuple(alternatives)
        if not candidates:
            return SemanticAbsorptionResult((), (), (), 0)
        alphabet_size = candidates[0].alphabet_size
        for automaton in candidates:
            if automaton.alphabet_size != alphabet_size:
                raise ValueError("cannot absorb DFAs over different alphabets")
            if any(
                output is not None and output is not True
                for output in automaton.outputs
            ):
                raise TypeError("semantic absorption requires pure acceptance outputs")

        count = len(candidates)
        included = [[False] * count for _ in range(count)]
        comparisons = 0
        for subset in range(count):
            included[subset][subset] = True
            for superset in range(count):
                if subset == superset:
                    continue
                comparisons += 1
                included[subset][superset] = acceptance_included(
                    candidates[subset], candidates[superset]
                )

        kept: list[int] = []
        for candidate in range(count):
            dominated = any(
                included[candidate][other]
                and (not included[other][candidate] or other < candidate)
                for other in range(count)
                if other != candidate
            )
            if not dominated:
                kept.append(candidate)

        absorbed_by: list[int | None] = [None] * count
        for candidate in range(count):
            if candidate in kept:
                continue
            dominators = [other for other in kept if included[candidate][other]]
            if not dominators:
                raise AssertionError(
                    "an absorbed language must have a maximal dominator"
                )
            absorbed_by[candidate] = min(dominators)
        return SemanticAbsorptionResult(
            tuple(candidates[index] for index in kept),
            tuple(kept),
            tuple(absorbed_by),
            comparisons,
        )


def absorb_acceptance_union(
    alternatives: Iterable[DFA[bool]],
) -> SemanticAbsorptionResult:
    return AutomatonSemanticAbsorber().run(alternatives)


__all__ = [
    "AutomatonSemanticAbsorber",
    "SemanticAbsorptionResult",
    "absorb_acceptance_union",
    "acceptance_included",
    "acceptance_inclusion_counterexample",
]
