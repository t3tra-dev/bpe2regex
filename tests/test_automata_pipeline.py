import itertools
import re
import unittest

from bpe2regex.reir import (
    DFA,
    AcceptanceAutomataCompiler,
    CostGuidedArdenEliminator,
    RegexCompiler,
    RegexSourceLowerer,
    SourceSizeCostModel,
    SymbolSet,
    Transition,
)


def _finite_words(accepted: set[bytes]) -> DFA[bool]:
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
        256,
        0,
        accepting,
        tuple(
            tuple(
                Transition(SymbolSet.singleton(256, symbol), target)
                for symbol, target in sorted(row.items())
            )
            for row in transitions
        ),
    )


def _lowerer() -> RegexSourceLowerer:
    return RegexSourceLowerer(escape_byte=lambda byte: f"\\x{byte:02x}")


class AcceptanceAutomataCompilerTests(unittest.TestCase):
    def test_pipeline_minimizes_absorbs_encodes_and_lowers(self) -> None:
        subset = _finite_words({b"a"})
        superset = _finite_words({b"a", b"b"})
        duplicate = _finite_words({b"a", b"b"})
        compiler = AcceptanceAutomataCompiler()
        result = compiler.run((subset, superset, duplicate))

        self.assertEqual(len(result.minimized_automata), 3)
        self.assertEqual(result.absorption.kept_indices, (1,))
        self.assertEqual(len(result.encoded_automata), 1)
        self.assertGreater(
            result.encoded_automata[0].default_transition_count,
            0,
        )

        source = RegexCompiler(_lowerer()).compile(result.expression)
        pattern = re.compile(source.encode("ascii"))
        for length in range(4):
            for values in itertools.product(b"abc", repeat=length):
                word = bytes(values)
                self.assertEqual(
                    pattern.fullmatch(word) is not None,
                    word in {b"a", b"b"},
                )

    def test_cost_guided_eliminator_can_be_injected(self) -> None:
        automaton = _finite_words({b"", b"ab", b"ba"})
        lowerer = _lowerer()
        eliminator = CostGuidedArdenEliminator(
            SourceSizeCostModel(lowerer),
            beam_width=8,
        )
        expression = AcceptanceAutomataCompiler(eliminator).compile((automaton,))
        pattern = re.compile(RegexCompiler(lowerer).compile(expression).encode("ascii"))
        for length in range(4):
            for values in itertools.product(b"ab", repeat=length):
                word = bytes(values)
                self.assertEqual(
                    pattern.fullmatch(word) is not None,
                    automaton.accepts(word),
                )

    def test_empty_union_lowers_to_the_empty_language(self) -> None:
        expression = AcceptanceAutomataCompiler().compile(())
        pattern = re.compile(
            RegexCompiler(_lowerer()).compile(expression).encode("ascii")
        )
        self.assertIsNone(pattern.fullmatch(b""))
        self.assertIsNone(pattern.fullmatch(b"a"))


if __name__ == "__main__":
    unittest.main()
