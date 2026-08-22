import itertools
import re
import unittest
from collections.abc import Hashable

from bpe2regex.reir import (
    DFA,
    ArdenEliminator,
    CostGuidedArdenEliminator,
    EliminationOrder,
    Op,
    RegexCompiler,
    RegexSourceLowerer,
    SourceSizeCostModel,
    SymbolSet,
    Transition,
)


class _FixedOrder(EliminationOrder):
    def __init__(self, order: tuple[int, ...]) -> None:
        self._order = order

    def order[OutputT: Hashable](
        self,
        automaton: DFA[OutputT],
        states: frozenset[int],
    ) -> tuple[int, ...]:
        return self._order


def _symbols(*symbols: int) -> SymbolSet:
    return SymbolSet.from_symbols(256, symbols)


def _lowerer() -> RegexSourceLowerer:
    return RegexSourceLowerer(escape_byte=lambda byte: f"\\x{byte:02x}")


def _source(expression: Op) -> str:
    return RegexCompiler(_lowerer()).compile(expression)


class EliminationOrderSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.automaton = DFA.accepting(
            256,
            0,
            (2,),
            (
                (
                    Transition(_symbols(ord("a")), 1),
                    Transition(_symbols(ord("b")), 2),
                ),
                (
                    Transition(_symbols(ord("a")), 0),
                    Transition(_symbols(ord("b")), 2),
                ),
                (
                    Transition(_symbols(ord("a")), 0),
                    Transition(_symbols(ord("b")), 1),
                ),
            ),
        )

    def test_wide_beam_finds_the_best_complete_scc_permutation(self) -> None:
        cost_model = SourceSizeCostModel(_lowerer())
        exhaustive = []
        for order in itertools.permutations(range(3)):
            expression = ArdenEliminator(_FixedOrder(order)).lower(self.automaton)
            cost = cost_model.evaluate(expression)
            exhaustive.append((cost_model.key(cost), order, expression))

        result = CostGuidedArdenEliminator(
            cost_model,
            beam_width=16,
        ).search(self.automaton)
        self.assertEqual(
            cost_model.key(result.cost), min(item[0] for item in exhaustive)
        )
        self.assertIn(result.order, tuple(item[1] for item in exhaustive))
        self.assertGreater(result.explored_candidates, 0)

    def test_search_result_preserves_language_and_is_deterministic(self) -> None:
        search = CostGuidedArdenEliminator(
            SourceSizeCostModel(_lowerer()),
            beam_width=4,
        )
        first = search.search(self.automaton)
        second = search.search(self.automaton)
        self.assertEqual(first, second)

        pattern = re.compile(_source(first.expression).encode("ascii"))
        for length in range(6):
            for values in itertools.product(b"ab", repeat=length):
                word = bytes(values)
                self.assertEqual(
                    pattern.fullmatch(word) is not None,
                    self.automaton.accepts(word),
                )

    def test_narrow_beam_never_loses_the_scc_baseline(self) -> None:
        cost_model = SourceSizeCostModel(_lowerer())
        baseline = ArdenEliminator().lower(self.automaton)
        result = CostGuidedArdenEliminator(cost_model, beam_width=1).search(
            self.automaton
        )
        self.assertLessEqual(
            cost_model.key(result.cost),
            cost_model.key(cost_model.evaluate(baseline)),
        )

    def test_invalid_beam_width_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            CostGuidedArdenEliminator(SourceSizeCostModel(_lowerer()), beam_width=0)


if __name__ == "__main__":
    unittest.main()
