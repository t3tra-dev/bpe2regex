import itertools
import re
import unittest

from bpe2regex.reir import (
    DFA,
    ArdenEliminator,
    DefaultTransition,
    RegexCompiler,
    RegexSourceLowerer,
    SymbolSet,
    Transition,
    encode_default_transitions,
    equivalent,
    expand_default_transitions,
)


def _symbols(alphabet_size: int, *symbols: int) -> SymbolSet:
    return SymbolSet.from_symbols(alphabet_size, symbols)


class DefaultTransitionTests(unittest.TestCase):
    def test_explicit_edges_override_the_remaining_default_alphabet(self) -> None:
        automaton = DFA(
            4,
            0,
            (None, "exception", "default"),
            (
                (Transition(_symbols(4, 1, 3), 1),),
                (),
                (),
            ),
            (
                DefaultTransition(2),
                DefaultTransition(1),
                DefaultTransition(2),
            ),
        )
        self.assertEqual(
            tuple(automaton.transition(0, symbol) for symbol in range(4)),
            (2, 1, 2, 1),
        )
        self.assertEqual(
            automaton.effective_transitions(0),
            (
                Transition(_symbols(4, 0, 2), 2),
                Transition(_symbols(4, 1, 3), 1),
            ),
        )
        self.assertTrue(automaton.is_total)

    def test_fully_shadowed_and_out_of_range_defaults_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shadowed"):
            DFA(
                2,
                0,
                (None,),
                ((Transition(SymbolSet.full(2), 0),),),
                (DefaultTransition(0),),
            )
        with self.assertRaisesRegex(ValueError, "out-of-range default"):
            DFA(2, 0, (None,), ((),), (DefaultTransition(1),))

    def test_dense_rows_round_trip_through_default_syntax(self) -> None:
        automaton = DFA(
            8,
            0,
            (None, True, None),
            (
                (
                    Transition(_symbols(8, 0, 1, 2, 3, 4, 5), 1),
                    Transition(_symbols(8, 6, 7), 2),
                ),
                (Transition(SymbolSet.full(8), 1),),
                (Transition(SymbolSet.full(8), 2),),
            ),
        )
        encoded = encode_default_transitions(automaton)
        self.assertEqual(encoded.default_transition_count, 3)
        self.assertEqual(encoded.defaults[0], DefaultTransition(1))
        self.assertEqual(
            encoded.transitions[0],
            (Transition(_symbols(8, 6, 7), 2),),
        )
        self.assertTrue(equivalent(automaton, encoded))
        self.assertEqual(expand_default_transitions(encoded), automaton)

    def test_partial_rows_are_not_accidentally_totalized(self) -> None:
        automaton = DFA.accepting(
            4,
            0,
            (1,),
            ((Transition(_symbols(4, 0), 1),), ()),
        )
        encoded = encode_default_transitions(automaton)
        self.assertEqual(encoded.defaults, (None, None))
        self.assertEqual(encoded, automaton)

    def test_default_syntax_survives_practical_byte_lowering(self) -> None:
        full = SymbolSet.full(256)
        explicit = DFA.accepting(
            256,
            0,
            (1,),
            (
                (
                    Transition(_symbols(256, ord("a")), 1),
                    Transition(full - _symbols(256, ord("a")), 0),
                ),
                (Transition(full, 1),),
            ),
        )
        encoded = encode_default_transitions(explicit)
        expression = ArdenEliminator().lower(encoded)
        source = RegexCompiler(
            RegexSourceLowerer(escape_byte=lambda byte: f"\\x{byte:02x}")
        ).compile(expression)
        pattern = re.compile(source.encode("ascii"))
        for length in range(4):
            for values in itertools.product(b"ab", repeat=length):
                word = bytes(values)
                self.assertEqual(
                    pattern.fullmatch(word) is not None,
                    explicit.accepts(word),
                )


if __name__ == "__main__":
    unittest.main()
