import itertools
import random
import unittest
from collections.abc import Iterable, Sequence

from bpe2regex.reir import (
    DFA,
    NEVER,
    CharSet,
    SymbolSet,
    Transition,
    alphabet_partition,
    coreachable_states,
    equivalence_counterexample,
    equivalent,
    minimize_dfa,
    prune_unreachable,
    reachable_states,
)


def _symbols(alphabet_size: int, *symbols: int) -> SymbolSet:
    return SymbolSet.from_symbols(alphabet_size, symbols)


def _row(
    alphabet_size: int,
    targets: Sequence[int | None],
) -> tuple[Transition, ...]:
    grouped: dict[int, list[int]] = {}
    for symbol, target in enumerate(targets):
        if target is not None:
            grouped.setdefault(target, []).append(symbol)
    return tuple(
        Transition(SymbolSet.from_symbols(alphabet_size, symbols), target)
        for target, symbols in grouped.items()
    )


def _words(alphabet_size: int, maximum_length: int) -> Iterable[tuple[int, ...]]:
    for length in range(maximum_length + 1):
        yield from itertools.product(range(alphabet_size), repeat=length)


class SymbolSetTests(unittest.TestCase):
    def test_bitset_algebra_and_intervals_are_canonical(self) -> None:
        left = _symbols(8, 0, 1, 3, 6, 7)
        right = _symbols(8, 1, 2, 3, 4)

        self.assertEqual(left.symbols, frozenset((0, 1, 3, 6, 7)))
        self.assertEqual(left.intervals, ((0, 1), (3, 3), (6, 7)))
        self.assertEqual((left | right).symbols, frozenset(range(8)) - {5})
        self.assertEqual((left & right).symbols, frozenset((1, 3)))
        self.assertEqual((left - right).symbols, frozenset((0, 6, 7)))
        self.assertEqual((left ^ right).symbols, frozenset((0, 2, 4, 6, 7)))
        self.assertEqual(left.complement().symbols, frozenset((2, 4, 5)))
        self.assertTrue(_symbols(8, 0, 1).issubset(left))
        self.assertTrue(_symbols(8, 2, 4).isdisjoint(left))

    def test_invalid_symbols_and_mismatched_alphabets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            SymbolSet.from_symbols(2, (2,))
        with self.assertRaisesRegex(TypeError, "integer"):
            SymbolSet.from_symbols(2, ("0",))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "different alphabets"):
            _symbols(2, 0).union(_symbols(3, 0))

    def test_byte_symbol_sets_bridge_to_pure_reir(self) -> None:
        charset = CharSet(b"az")
        self.assertEqual(SymbolSet.from_charset(charset).to_reir(), charset)
        self.assertIs(SymbolSet.empty(256).to_reir(), NEVER)
        with self.assertRaisesRegex(ValueError, "byte alphabet"):
            _symbols(2, 0).to_reir()


class DFAInvariantAndExecutionTests(unittest.TestCase):
    def test_constructor_merges_same_target_labels_and_sorts_rows(self) -> None:
        automaton = DFA(
            4,
            0,
            (None, True),
            (
                (
                    Transition(_symbols(4, 2), 1),
                    Transition(_symbols(4, 0, 1), 1),
                ),
                (),
            ),
        )
        self.assertEqual(
            automaton.transitions[0],
            (Transition(_symbols(4, 0, 1, 2), 1),),
        )

    def test_overlapping_different_targets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlapping"):
            DFA(
                2,
                0,
                (None, True),
                (
                    (
                        Transition(_symbols(2, 0, 1), 0),
                        Transition(_symbols(2, 1), 1),
                    ),
                    (),
                ),
            )

    def test_partial_execution_trace_and_totalization(self) -> None:
        automaton = DFA.accepting(
            2,
            0,
            (1,),
            (
                (Transition(_symbols(2, 0), 1),),
                (),
            ),
        )
        self.assertFalse(automaton.accepts(()))
        self.assertTrue(automaton.accepts((0,)))
        self.assertFalse(automaton.accepts((1,)))
        self.assertEqual(automaton.trace((0,)), (0, 1))
        self.assertIsNone(automaton.trace((1,)))
        self.assertFalse(automaton.is_total)

        total = automaton.totalize()
        self.assertTrue(total.is_total)
        self.assertEqual(total.state_count, 3)
        for word in _words(2, 4):
            self.assertEqual(total.match_output(word), automaton.match_output(word))


class AutomatonAnalysisTests(unittest.TestCase):
    def test_alphabet_partition_groups_symbols_by_global_behavior(self) -> None:
        automaton = DFA(
            4,
            0,
            (None, True),
            (
                (
                    Transition(_symbols(4, 0, 1), 1),
                    Transition(_symbols(4, 2), 0),
                ),
                (
                    Transition(_symbols(4, 0, 1), 1),
                    Transition(_symbols(4, 2), 0),
                ),
            ),
        )
        self.assertEqual(
            tuple(group.symbols for group in alphabet_partition(automaton)),
            (frozenset((0, 1)), frozenset((2,)), frozenset((3,))),
        )

    def test_reachability_coreachability_and_pruning(self) -> None:
        full = SymbolSet.full(2)
        automaton = DFA(
            2,
            0,
            (None, True, None, None),
            (
                (
                    Transition(_symbols(2, 0), 1),
                    Transition(_symbols(2, 1), 2),
                ),
                (Transition(full, 1),),
                (Transition(full, 2),),
                (Transition(full, 1),),
            ),
        )
        self.assertEqual(reachable_states(automaton), (0, 1, 2))
        self.assertEqual(coreachable_states(automaton), frozenset((0, 1, 3)))

        pruned = prune_unreachable(automaton)
        self.assertEqual(pruned.state_map, (0, 1, 2, None))
        self.assertEqual(
            pruned.blocks, (frozenset((0,)), frozenset((1,)), frozenset((2,)))
        )
        self.assertEqual(pruned.automaton.state_count, 3)


class DFAMinimizationTests(unittest.TestCase):
    def test_equivalent_residual_states_are_quotiented_and_reindexed(self) -> None:
        full = SymbolSet.full(2)
        automaton = DFA(
            2,
            0,
            (None, True, True, None),
            (
                (
                    Transition(_symbols(2, 0), 1),
                    Transition(_symbols(2, 1), 2),
                ),
                (Transition(full, 1),),
                (Transition(full, 2),),
                (Transition(full, 3),),
            ),
        )
        minimized = minimize_dfa(automaton)

        self.assertEqual(minimized.automaton.state_count, 2)
        self.assertTrue(minimized.automaton.is_total)
        self.assertEqual(minimized.state_map, (0, 1, 1, None))
        self.assertEqual(minimized.blocks, (frozenset((0,)), frozenset((1, 2))))
        self.assertTrue(equivalent(automaton, minimized.automaton))
        self.assertEqual(
            minimize_dfa(minimized.automaton).automaton,
            minimized.automaton,
        )

    def test_observably_different_outputs_are_not_merged(self) -> None:
        full = SymbolSet.full(2)
        automaton = DFA(
            2,
            0,
            (None, "left", "right"),
            (
                (
                    Transition(_symbols(2, 0), 1),
                    Transition(_symbols(2, 1), 2),
                ),
                (Transition(full, 1),),
                (Transition(full, 2),),
            ),
        )
        minimized = minimize_dfa(automaton)
        self.assertEqual(minimized.automaton.state_count, 3)
        self.assertNotEqual(minimized.state_map[1], minimized.state_map[2])

    def test_partial_automaton_is_minimized_with_implicit_reject_sink(self) -> None:
        automaton = DFA.accepting(
            2,
            0,
            (1,),
            (
                (Transition(_symbols(2, 0), 1),),
                (),
            ),
        )
        minimized = minimize_dfa(automaton).automaton
        self.assertEqual(minimized.state_count, 3)
        self.assertTrue(minimized.is_total)
        for word in _words(2, 5):
            self.assertEqual(minimized.match_output(word), automaton.match_output(word))

    def test_random_partial_dfas_preserve_all_short_outputs(self) -> None:
        randomizer = random.Random(20_260_822)
        for _ in range(100):
            alphabet_size = randomizer.randrange(2, 4)
            state_count = randomizer.randrange(1, 7)
            outputs = tuple(
                randomizer.choice((None, None, 0, 1)) for _ in range(state_count)
            )
            rows = tuple(
                _row(
                    alphabet_size,
                    tuple(
                        randomizer.choice((*range(state_count), None))
                        for _ in range(alphabet_size)
                    ),
                )
                for _ in range(state_count)
            )
            automaton = DFA(alphabet_size, 0, outputs, rows)
            minimized = minimize_dfa(automaton).automaton
            self.assertTrue(equivalent(automaton, minimized))
            self.assertEqual(minimize_dfa(minimized).automaton, minimized)
            for word in _words(alphabet_size, 5):
                self.assertEqual(
                    minimized.match_output(word),
                    automaton.match_output(word),
                    (automaton, minimized, word),
                )

    def test_all_two_state_binary_partial_dfas_preserve_language(self) -> None:
        words = tuple(_words(2, 4))
        for outputs in itertools.product((None, True), repeat=2):
            for targets in itertools.product((None, 0, 1), repeat=4):
                automaton = DFA(
                    2,
                    0,
                    outputs,
                    (
                        _row(2, targets[:2]),
                        _row(2, targets[2:]),
                    ),
                )
                minimized = minimize_dfa(automaton).automaton
                self.assertTrue(equivalent(automaton, minimized))
                for word in words:
                    self.assertEqual(
                        minimized.match_output(word),
                        automaton.match_output(word),
                        (automaton, minimized, word),
                    )


class DFAEquivalenceTests(unittest.TestCase):
    def test_counterexample_is_shortest_and_lexicographically_first(self) -> None:
        left = DFA.accepting(
            2,
            0,
            (1,),
            ((Transition(_symbols(2, 0), 1),), ()),
        )
        right = DFA.accepting(
            2,
            0,
            (1,),
            ((Transition(_symbols(2, 1), 1),), ()),
        )
        self.assertEqual(equivalence_counterexample(left, right), (0,))
        self.assertFalse(equivalent(left, right))

    def test_empty_word_can_be_a_counterexample(self) -> None:
        accepting = DFA.accepting(2, 0, (0,), ((),))
        rejecting = DFA.accepting(2, 0, (), ((),))
        self.assertEqual(equivalence_counterexample(accepting, rejecting), ())

    def test_different_alphabets_cannot_be_compared(self) -> None:
        left = DFA.accepting(2, 0, (), ((),))
        right = DFA.accepting(3, 0, (), ((),))
        with self.assertRaisesRegex(ValueError, "different alphabets"):
            equivalence_counterexample(left, right)


if __name__ == "__main__":
    unittest.main()
