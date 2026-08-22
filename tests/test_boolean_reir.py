import itertools
import random
import re
import unittest
from collections.abc import Iterable

from bpe2regex.reir import (
    EPSILON,
    NEVER,
    UNIVERSAL,
    BooleanDerivativeDFACompiler,
    BooleanDerivativeEngine,
    BooleanDerivativeStateBudgetExceeded,
    Complement,
    Difference,
    Intersect,
    Op,
    RegexCompiler,
    RegexSourceLowerer,
    alternate,
    charset,
    compile_boolean_dfa,
    complement,
    concat,
    contains_boolean,
    difference,
    equivalent,
    intersect,
    literal,
    lower_boolean_ops_to_core,
    lower_boolean_to_core,
    nullable,
    repeat,
)


def _words(symbols: bytes, maximum_length: int) -> Iterable[bytes]:
    for length in range(maximum_length + 1):
        yield from map(bytes, itertools.product(symbols, repeat=length))


def _compile_core(expression: Op) -> re.Pattern[bytes]:
    source = RegexCompiler(
        RegexSourceLowerer(escape_byte=lambda byte: f"\\x{byte:02x}")
    ).compile(expression)
    return re.compile(source.encode("ascii"))


def _random_core(randomizer: random.Random, depth: int) -> Op:
    leaves = (NEVER, EPSILON, literal(b"a"), literal(b"b"), charset(b"ab"))
    if depth == 0 or randomizer.randrange(4) == 0:
        return randomizer.choice(leaves)
    choice = randomizer.randrange(3)
    if choice == 0:
        return concat(
            _random_core(randomizer, depth - 1),
            _random_core(randomizer, depth - 1),
        )
    if choice == 1:
        return alternate(
            _random_core(randomizer, depth - 1),
            _random_core(randomizer, depth - 1),
        )
    return repeat(
        _random_core(randomizer, depth - 1),
        *randomizer.choice(((0, 1), (0, 2), (0, None), (1, 3))),
    )


def _random_boolean(randomizer: random.Random, depth: int) -> Op:
    if depth == 0 or randomizer.randrange(3) == 0:
        return _random_core(randomizer, 2)
    left = _random_boolean(randomizer, depth - 1)
    choice = randomizer.randrange(3)
    if choice == 0:
        return complement(left)
    right = _random_boolean(randomizer, depth - 1)
    return intersect(left, right) if choice == 1 else difference(left, right)


class BooleanBuilderTests(unittest.TestCase):
    def test_complement_and_difference_canonicalization(self) -> None:
        value = literal(b"a")
        self.assertIs(complement(NEVER), UNIVERSAL)
        self.assertIs(complement(UNIVERSAL), NEVER)
        self.assertEqual(complement(complement(value)), value)
        self.assertIs(difference(value, value), NEVER)
        self.assertEqual(difference(value, NEVER), value)
        self.assertEqual(difference(UNIVERSAL, value), complement(value))
        self.assertEqual(
            difference(value, complement(literal(b"b"))),
            intersect(value, literal(b"b")),
        )

    def test_intersection_is_flat_sorted_unique_and_absorbing(self) -> None:
        left = literal(b"a")
        right = literal(b"b")
        self.assertEqual(
            intersect(right, intersect(left, right)), intersect(left, right)
        )
        self.assertEqual(intersect(UNIVERSAL, left), left)
        self.assertIs(intersect(NEVER, left), NEVER)
        self.assertIs(intersect(left, complement(left)), NEVER)
        self.assertIs(intersect(), UNIVERSAL)

    def test_language_complement_is_not_a_character_class_complement(self) -> None:
        expression = complement(charset(b"a"))
        self.assertIsInstance(expression, Complement)
        automaton = compile_boolean_dfa(expression)
        self.assertTrue(automaton.accepts(b""))
        self.assertFalse(automaton.accepts(b"a"))
        self.assertTrue(automaton.accepts(b"b"))
        self.assertTrue(automaton.accepts(b"aa"))


class BooleanDerivativeTests(unittest.TestCase):
    def test_nullability_rules(self) -> None:
        self.assertTrue(nullable(UNIVERSAL))
        self.assertTrue(nullable(complement(literal(b"a"))))
        self.assertFalse(nullable(complement(EPSILON)))
        self.assertFalse(nullable(intersect(EPSILON, literal(b"a"))))
        self.assertTrue(nullable(difference(EPSILON, literal(b"a"))))
        self.assertFalse(nullable(difference(EPSILON, EPSILON)))

    def test_all_256_derivatives_follow_boolean_algebra(self) -> None:
        left = alternate(
            *(literal(bytes((symbol, symbol ^ 0xFF))) for symbol in range(256))
        )
        right = charset(range(0, 256, 2))
        expression = difference(left, right)
        engine = BooleanDerivativeEngine()
        for symbol in range(256):
            with self.subTest(symbol=symbol):
                expected = (
                    literal(bytes((symbol ^ 0xFF,)))
                    if symbol % 2
                    else difference(literal(bytes((symbol ^ 0xFF,))), EPSILON)
                )
                self.assertEqual(engine.derive(expression, symbol), expected)
        self.assertGreaterEqual(engine.cached_derivative_count, 256 * 3)

    def test_nullable_concat_and_repeat_are_differentiated_exactly(self) -> None:
        nullable_body = complement(literal(b"a"))
        expression = concat(
            repeat(nullable_body, 2, 4),
            difference(charset(b"ab"), literal(b"a")),
        )
        derivative = BooleanDerivativeEngine().derive(expression, ord("b"))
        original_dfa = compile_boolean_dfa(expression)
        derivative_dfa = compile_boolean_dfa(derivative)
        for suffix in _words(b"ab", 4):
            with self.subTest(suffix=suffix):
                self.assertEqual(
                    original_dfa.accepts(b"b" + suffix),
                    derivative_dfa.accepts(suffix),
                )


class BooleanAutomataAndLoweringTests(unittest.TestCase):
    def test_derivative_residuals_are_grouped_into_symbol_sets(self) -> None:
        expression = complement(charset(b"ab"))
        result = BooleanDerivativeDFACompiler().compile(expression)
        self.assertEqual(result.automaton.state_count, 3)
        self.assertEqual(len(result.automaton.transitions[0]), 2)
        grouped = {
            transition.symbols.symbols: result.residuals[transition.target]
            for transition in result.automaton.transitions[0]
        }
        self.assertEqual(grouped[frozenset(b"ab")], complement(EPSILON))
        self.assertEqual(
            grouped[frozenset(range(256)) - frozenset(b"ab")],
            UNIVERSAL,
        )

    def test_derivative_closure_obeys_the_state_budget(self) -> None:
        with self.assertRaises(BooleanDerivativeStateBudgetExceeded) as raised:
            BooleanDerivativeDFACompiler(max_states=1).compile(
                complement(literal(b"a"))
            )
        self.assertEqual(raised.exception.max_states, 1)

    def test_core_lowering_preserves_small_alphabet_language(self) -> None:
        expressions = (
            complement(alternate(literal(b"a"), literal(b"ab"))),
            intersect(repeat(charset(b"ab"), 1, 3), complement(literal(b"aa"))),
            difference(repeat(charset(b"ab"), 0, None), literal(b"ba")),
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                core = lower_boolean_to_core(expression)
                self.assertFalse(contains_boolean(core))
                original = compile_boolean_dfa(expression)
                lowered = compile_boolean_dfa(core)
                self.assertTrue(equivalent(original, lowered))
                pattern = _compile_core(core)
                for word in _words(b"ab", 5):
                    self.assertEqual(
                        original.accepts(word),
                        pattern.fullmatch(word) is not None,
                    )

    def test_randomized_boolean_languages_survive_core_lowering(self) -> None:
        randomizer = random.Random(0xB001)
        for _ in range(40):
            expression = _random_boolean(randomizer, 3)
            core = lower_boolean_to_core(expression, max_states=500)
            self.assertTrue(
                equivalent(
                    compile_boolean_dfa(expression, max_states=500),
                    compile_boolean_dfa(core, max_states=500),
                )
            )

    def test_boolean_node_shapes_remain_distinct_from_core_seven_ops(self) -> None:
        left = literal(b"a")
        right = literal(b"b")
        self.assertIsInstance(intersect(left, right), Intersect)
        self.assertIsInstance(difference(left, right), Difference)
        self.assertTrue(contains_boolean(complement(left)))

    def test_local_conversion_preserves_surrounding_core_structure(self) -> None:
        expression = concat(
            literal(b"prefix"),
            difference(charset(b"ab"), literal(b"a")),
            literal(b"suffix"),
        )
        lowered = lower_boolean_ops_to_core(expression)
        self.assertFalse(contains_boolean(lowered))
        self.assertTrue(
            equivalent(compile_boolean_dfa(expression), compile_boolean_dfa(lowered))
        )


if __name__ == "__main__":
    unittest.main()
