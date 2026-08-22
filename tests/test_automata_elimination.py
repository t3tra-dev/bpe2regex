import itertools
import random
import re
import unittest

from bpe2regex.reir import (
    DFA,
    NEVER,
    ArdenEliminator,
    Op,
    RegexCompiler,
    RegexSourceLowerer,
    SCCEliminationOrder,
    SymbolSet,
    Transition,
    strongly_connected_components,
)


def _symbols(*symbols: int) -> SymbolSet:
    return SymbolSet.from_symbols(256, symbols)


def _row(targets: tuple[int | None, ...]) -> tuple[Transition, ...]:
    grouped: dict[int, list[int]] = {}
    for symbol, target in zip(b"ab", targets, strict=True):
        if target is not None:
            grouped.setdefault(target, []).append(symbol)
    return tuple(
        Transition(SymbolSet.from_symbols(256, symbols), target)
        for target, symbols in grouped.items()
    )


def _compile(expression: Op) -> re.Pattern[bytes]:
    compiler = RegexCompiler(
        RegexSourceLowerer(escape_byte=lambda byte: f"\\x{byte:02x}")
    )
    return re.compile(compiler.compile(expression).encode("ascii"))


def _words(maximum_length: int) -> tuple[bytes, ...]:
    return tuple(
        bytes(values)
        for length in range(maximum_length + 1)
        for values in itertools.product(b"ab", repeat=length)
    )


class SCCAnalysisTests(unittest.TestCase):
    def test_tarjan_components_are_reverse_topological_and_deterministic(self) -> None:
        automaton = DFA.accepting(
            256,
            0,
            (4,),
            (
                (Transition(_symbols(ord("a")), 1),),
                (Transition(_symbols(ord("a")), 2),),
                (Transition(_symbols(ord("a")), 1), Transition(_symbols(ord("b")), 3)),
                (Transition(_symbols(ord("a")), 4),),
                (),
            ),
        )
        self.assertEqual(
            strongly_connected_components(automaton),
            (
                frozenset((4,)),
                frozenset((3,)),
                frozenset((1, 2)),
                frozenset((0,)),
            ),
        )
        self.assertEqual(
            SCCEliminationOrder().order(automaton, frozenset(range(5))),
            (4, 3, 1, 2, 0),
        )

    def test_scc_analysis_does_not_depend_on_python_recursion_depth(self) -> None:
        state_count = 1_501
        symbol = SymbolSet.singleton(1, 0)
        automaton = DFA.accepting(
            1,
            0,
            (state_count - 1,),
            tuple(
                (Transition(symbol, state + 1),) if state + 1 < state_count else ()
                for state in range(state_count)
            ),
        )
        components = strongly_connected_components(automaton)
        self.assertEqual(len(components), state_count)
        self.assertEqual(components[0], frozenset((state_count - 1,)))
        self.assertEqual(components[-1], frozenset((0,)))


class ArdenEliminationTests(unittest.TestCase):
    def test_loop_and_chain_lower_to_the_same_language(self) -> None:
        automaton = DFA.accepting(
            256,
            0,
            (2,),
            (
                (Transition(_symbols(ord("a")), 1),),
                (
                    Transition(_symbols(ord("b")), 1),
                    Transition(_symbols(ord("c")), 2),
                ),
                (),
            ),
        )
        pattern = _compile(ArdenEliminator().lower(automaton))
        for value in (b"ac", b"abc", b"abbbc"):
            self.assertIsNotNone(pattern.fullmatch(value))
        for value in (b"", b"a", b"bc", b"abbd"):
            self.assertIsNone(pattern.fullmatch(value))

    def test_nontrivial_scc_is_solved_with_arden_updates(self) -> None:
        automaton = DFA.accepting(
            256,
            0,
            (2,),
            (
                (
                    Transition(_symbols(ord("a")), 1),
                    Transition(_symbols(ord("c")), 2),
                ),
                (Transition(_symbols(ord("b")), 0),),
                (),
            ),
        )
        pattern = _compile(ArdenEliminator().lower(automaton))
        for value in (b"c", b"abc", b"abababc"):
            self.assertIsNotNone(pattern.fullmatch(value))
        for value in (b"", b"a", b"ab", b"ac"):
            self.assertIsNone(pattern.fullmatch(value))

    def test_output_languages_remain_separate(self) -> None:
        automaton = DFA(
            256,
            0,
            (None, "left", "right"),
            (
                (
                    Transition(_symbols(ord("a")), 1),
                    Transition(_symbols(ord("b")), 2),
                ),
                (),
                (),
            ),
        )
        languages = ArdenEliminator().lower_outputs(automaton)
        self.assertEqual(tuple(item.output for item in languages), ("left", "right"))
        self.assertIsNotNone(_compile(languages[0].expression).fullmatch(b"a"))
        self.assertIsNone(_compile(languages[0].expression).fullmatch(b"b"))
        self.assertIsNotNone(_compile(languages[1].expression).fullmatch(b"b"))
        self.assertIsNone(_compile(languages[1].expression).fullmatch(b"a"))

    def test_empty_or_non_byte_languages_are_rejected_cleanly(self) -> None:
        empty = DFA.accepting(256, 0, (), ((),))
        self.assertIs(ArdenEliminator().lower(empty), NEVER)
        with self.assertRaisesRegex(ValueError, "reserved rejecting"):
            ArdenEliminator().lower_output(empty, None)  # type: ignore[arg-type]
        non_byte = DFA.accepting(2, 0, (), ((),))
        with self.assertRaisesRegex(ValueError, "byte-alphabet"):
            ArdenEliminator().lower(non_byte)

    def test_random_small_dfas_match_lowered_reir_exhaustively(self) -> None:
        randomizer = random.Random(20_260_822)
        words = _words(5)
        for _ in range(75):
            state_count = randomizer.randrange(1, 6)
            automaton = DFA.accepting(
                256,
                0,
                (state for state in range(state_count) if randomizer.randrange(3) == 0),
                tuple(
                    _row(
                        (
                            randomizer.choice((*range(state_count), None)),
                            randomizer.choice((*range(state_count), None)),
                        )
                    )
                    for _ in range(state_count)
                ),
            )
            pattern = _compile(ArdenEliminator().lower(automaton))
            for word in words:
                self.assertEqual(
                    pattern.fullmatch(word) is not None,
                    automaton.accepts(word),
                    (automaton, word),
                )


if __name__ == "__main__":
    unittest.main()
