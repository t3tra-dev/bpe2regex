import itertools
import random
import re
import unittest

from bpe2regex.reir import (
    EPSILON,
    NEVER,
    CharSet,
    DerivativeEngine,
    Op,
    RegexCompiler,
    RegexSourceLowerer,
    Repeat,
    alternate,
    charset,
    concat,
    derivative,
    group_derivatives,
    literal,
    repeat,
)
from bpe2regex.reir.tagged import tagged


def _byte_escape(byte: int) -> str:
    return f"\\x{byte:02x}"


def _compile(expression: Op) -> re.Pattern[bytes]:
    compiler = RegexCompiler(RegexSourceLowerer(escape_byte=_byte_escape))
    return re.compile(compiler.compile(expression).encode("ascii"))


class DerivativeRuleTests(unittest.TestCase):
    def test_leaf_rules_cover_never_epsilon_charset_and_literal(self) -> None:
        engine = DerivativeEngine()
        self.assertIs(engine.derive(NEVER, ord("a")), NEVER)
        self.assertIs(engine.derive(EPSILON, ord("a")), NEVER)
        self.assertIs(engine.derive(charset(b"ab"), ord("a")), EPSILON)
        self.assertIs(engine.derive(charset(b"ab"), ord("c")), NEVER)
        self.assertEqual(engine.derive(literal(b"abc"), ord("a")), literal(b"bc"))
        self.assertIs(engine.derive(literal(b"abc"), ord("b")), NEVER)

    def test_alternate_and_nullable_concat_rules(self) -> None:
        engine = DerivativeEngine()
        branch = alternate(literal(b"ab"), literal(b"ac"))
        self.assertEqual(engine.derive(branch, ord("a")), charset(b"bc"))

        nullable_head = alternate(EPSILON, literal(b"a"))
        expression = concat(nullable_head, literal(b"b"))
        self.assertEqual(engine.derive(expression, ord("a")), literal(b"b"))
        self.assertIs(engine.derive(expression, ord("b")), EPSILON)

        overlapping = concat(nullable_head, literal(b"a"))
        self.assertEqual(
            engine.derive(overlapping, ord("a")),
            alternate(EPSILON, literal(b"a")),
        )

    def test_repeat_rules_cover_bounded_unbounded_and_zero_upper_bound(self) -> None:
        engine = DerivativeEngine()
        body = literal(b"a")
        self.assertEqual(
            engine.derive(Repeat(body, 2, 4), ord("a")),
            repeat(body, 1, 3),
        )
        self.assertIs(engine.derive(Repeat(body, 0, 1), ord("a")), EPSILON)
        self.assertEqual(
            engine.derive(Repeat(body, 0, None), ord("a")),
            repeat(body, 0, None),
        )
        self.assertIs(engine.derive(Repeat(body, 0, 0), ord("a")), NEVER)

    def test_nullable_repeat_uses_zero_as_the_residual_lower_bound(self) -> None:
        engine = DerivativeEngine()
        body = alternate(EPSILON, literal(b"a"))
        self.assertEqual(
            engine.derive(Repeat(body, 2, 4), ord("a")),
            repeat(body, 0, 3),
        )
        self.assertEqual(
            engine.derive(Repeat(body, 5, None), ord("a")),
            repeat(body, 0, None),
        )

    def test_nullable_repeat_derivatives_are_left_quotients_for_all_bound_forms(
        self,
    ) -> None:
        body = alternate(EPSILON, literal(b"a"), literal(b"ab"))
        suffixes = tuple(
            bytes(values)
            for length in range(5)
            for values in itertools.product(b"ab", repeat=length)
        )
        for minimum, maximum in ((0, 0), (0, 3), (2, 4), (5, None)):
            expression = Repeat(body, minimum, maximum)
            before = _compile(expression)
            engine = DerivativeEngine()
            for symbol in b"ab":
                after = _compile(engine.derive(expression, symbol))
                for suffix in suffixes:
                    with self.subTest(
                        bounds=(minimum, maximum),
                        symbol=symbol,
                        suffix=suffix,
                    ):
                        self.assertEqual(
                            before.fullmatch(bytes((symbol,)) + suffix) is not None,
                            after.fullmatch(suffix) is not None,
                        )

    def test_derivatives_are_memoized_by_operation_identity_and_byte(self) -> None:
        engine = DerivativeEngine()
        expression = literal(b"abc")
        first = engine.derive(expression, ord("a"))
        cache_size = engine.cached_derivative_count
        second = engine.derive(expression, ord("a"))
        self.assertIs(second, first)
        self.assertEqual(engine.cached_derivative_count, cache_size)
        self.assertEqual(engine.cached_symbols(expression), frozenset((ord("a"),)))

    def test_invalid_symbols_and_tagged_semantics_are_rejected(self) -> None:
        engine = DerivativeEngine()
        with self.assertRaisesRegex(ValueError, "byte alphabet"):
            engine.derive(literal(b"a"), 256)
        with self.assertRaisesRegex(TypeError, "integer byte"):
            engine.derive(literal(b"a"), "a")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "pure regex"):
            engine.derive(tagged(1), ord("a"))


class DerivativeGroupingTests(unittest.TestCase):
    def test_first_bytes_are_grouped_by_equal_residual(self) -> None:
        expression = alternate(
            literal(b"ax"),
            literal(b"bx"),
            literal(b"cy"),
        )
        engine = DerivativeEngine()
        groups = engine.group(expression)
        self.assertEqual(
            tuple((group.symbols.symbols, group.residual) for group in groups),
            (
                (frozenset(b"ab"), literal(b"x")),
                (frozenset(b"c"), literal(b"y")),
            ),
        )
        self.assertEqual(engine.cached_symbols(expression), frozenset(b"abc"))

    def test_grouping_evaluates_at_most_the_256_byte_alphabet(self) -> None:
        expression = charset(range(256))
        engine = DerivativeEngine()
        groups = engine.group(expression)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].symbols, CharSet(range(256)))
        self.assertIs(groups[0].residual, EPSILON)
        self.assertEqual(engine.cached_derivative_count, 256)
        self.assertEqual(engine.cached_symbols(expression), frozenset(range(256)))

    def test_convenience_functions_share_the_engine_semantics(self) -> None:
        expression = alternate(literal(b"ax"), literal(b"bx"))
        self.assertEqual(derivative(expression, ord("a")), literal(b"x"))
        groups = group_derivatives(expression)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].symbols.symbols, frozenset(b"ab"))
        self.assertEqual(groups[0].residual, literal(b"x"))


class DerivativeLanguageEquivalenceTests(unittest.TestCase):
    def test_random_derivatives_are_left_quotients(self) -> None:
        randomizer = random.Random(20_260_821)
        leaves = (
            EPSILON,
            NEVER,
            literal(b"a"),
            literal(b"b"),
            literal(b"ab"),
            charset(b"ab"),
        )

        def generate(depth: int) -> Op:
            if depth == 0 or randomizer.randrange(4) == 0:
                return randomizer.choice(leaves)
            operation = randomizer.randrange(3)
            if operation == 0:
                return concat(*(generate(depth - 1) for _ in range(2)))
            if operation == 1:
                return alternate(*(generate(depth - 1) for _ in range(3)))
            minimum, maximum = randomizer.choice(
                ((0, 1), (0, 2), (0, None), (1, 3), (2, 4))
            )
            return repeat(generate(depth - 1), minimum, maximum)

        suffixes = tuple(
            bytes(values)
            for length in range(4)
            for values in itertools.product(b"ab", repeat=length)
        )
        for _ in range(100):
            expression = generate(3)
            before = _compile(expression)
            engine = DerivativeEngine()
            for symbol in b"abc":
                after = _compile(engine.derive(expression, symbol))
                for suffix in suffixes:
                    self.assertEqual(
                        before.fullmatch(bytes((symbol,)) + suffix) is not None,
                        after.fullmatch(suffix) is not None,
                        (expression, symbol, suffix),
                    )


if __name__ == "__main__":
    unittest.main()
