import itertools
import unittest

from bpe2regex.reir import (
    DFA,
    AutomatonSemanticAbsorber,
    DefaultTransition,
    SymbolSet,
    Transition,
    absorb_acceptance_union,
    acceptance_included,
    acceptance_inclusion_counterexample,
)


def _symbols(alphabet_size: int, *symbols: int) -> SymbolSet:
    return SymbolSet.from_symbols(alphabet_size, symbols)


def _finite_words(alphabet_size: int, accepted: set[tuple[int, ...]]) -> DFA[bool]:
    transitions: list[dict[int, int]] = [{}]
    accepting: set[int] = set()
    for word in sorted(accepted):
        state = 0
        for symbol in word:
            target = transitions[state].get(symbol)
            if target is None:
                target = len(transitions)
                transitions[state][symbol] = target
                transitions.append({})
            state = target
        accepting.add(state)
    return DFA.accepting(
        alphabet_size,
        0,
        accepting,
        tuple(
            tuple(
                Transition(_symbols(alphabet_size, symbol), target)
                for symbol, target in sorted(row.items())
            )
            for row in transitions
        ),
    )


class AcceptanceInclusionTests(unittest.TestCase):
    def test_counterexample_is_shortest_and_lexicographically_first(self) -> None:
        subset = _finite_words(2, {(0,), (1,)})
        superset = _finite_words(2, {(1,)})
        self.assertEqual(
            acceptance_inclusion_counterexample(subset, superset),
            (0,),
        )
        self.assertFalse(acceptance_included(subset, superset))
        self.assertTrue(acceptance_included(superset, subset))

    def test_default_transition_semantics_participate_in_inclusion(self) -> None:
        all_single_symbols = DFA.accepting(
            2,
            0,
            (1,),
            ((), ()),
            (DefaultTransition(1), None),
        )
        only_zero = _finite_words(2, {(0,)})
        self.assertTrue(acceptance_included(only_zero, all_single_symbols))
        self.assertEqual(
            acceptance_inclusion_counterexample(all_single_symbols, only_zero),
            (1,),
        )


class AutomatonSemanticAbsorptionTests(unittest.TestCase):
    def test_subset_and_later_equivalent_alternatives_are_absorbed(self) -> None:
        only_zero = _finite_words(2, {(0,)})
        both = _finite_words(2, {(0,), (1,)})
        duplicate = _finite_words(2, {(0,), (1,)})

        result = absorb_acceptance_union((only_zero, both, duplicate))
        self.assertEqual(result.alternatives, (both,))
        self.assertEqual(result.kept_indices, (1,))
        self.assertEqual(result.absorbed_by, (1, None, 1))
        self.assertEqual(result.comparison_count, 6)

        for length in range(4):
            for word in itertools.product(range(2), repeat=length):
                before = any(
                    automaton.accepts(word)
                    for automaton in (only_zero, both, duplicate)
                )
                after = any(
                    automaton.accepts(word) for automaton in result.alternatives
                )
                self.assertEqual(before, after)

    def test_incomparable_languages_are_both_retained(self) -> None:
        zero = _finite_words(2, {(0,)})
        one = _finite_words(2, {(1,)})
        result = AutomatonSemanticAbsorber().run((zero, one))
        self.assertEqual(result.kept_indices, (0, 1))
        self.assertEqual(result.absorbed_by, (None, None))

    def test_tagged_outputs_and_mixed_alphabets_are_rejected(self) -> None:
        tagged = DFA(2, 0, (7,), ((),))
        with self.assertRaisesRegex(TypeError, "pure acceptance"):
            AutomatonSemanticAbsorber().run((tagged,))  # type: ignore[arg-type]

        binary = _finite_words(2, set())
        ternary = _finite_words(3, set())
        with self.assertRaisesRegex(ValueError, "different alphabets"):
            AutomatonSemanticAbsorber().run((binary, ternary))


if __name__ == "__main__":
    unittest.main()
