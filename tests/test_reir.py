import inspect
import itertools
import random
import re
import unittest
from dataclasses import dataclass

from bpe2regex.reir import (
    EPSILON,
    NEVER,
    Alternate,
    AnalysisManager,
    CandidateGenerator,
    CandidateSelectionPass,
    CandidateSelector,
    CanonicalizePass,
    CharSet,
    Concat,
    CostModel,
    DataFlowAnalysis,
    DeflatedSourceCostModel,
    FunctionalCandidateGenerator,
    FunctionalCostModel,
    GreedyRewriteDriver,
    Literal,
    LoweredSizeCost,
    LoweredSizeCostModel,
    Lowerer,
    LoweringContext,
    MinimumCostSelector,
    Op,
    OperationPass,
    OpLowerer,
    PassManager,
    PatternRewriter,
    PureOp,
    RegexCompiler,
    RegexPropertiesAnalysis,
    RegexSourceLowerer,
    Repeat,
    RewritePattern,
    SourceSizeCostModel,
    StructuralCost,
    StructuralCostModel,
    StructureDiscoveryPass,
    alternate,
    benchmark_compiler,
    charset,
    concat,
    literal,
    raw_deflate_size,
    render_regex,
    repeat,
)
from bpe2regex.reir.tagged import (
    TAGGED_BUILDER,
    TaggedAlternate,
    TaggedConcat,
    tagged,
)
from bpe2regex.reir.tagged_source import render_tagged_regex


def _ascii_escape(byte: int) -> str:
    return (
        chr(byte) if chr(byte).isascii() and chr(byte).isalnum() else f"\\x{byte:02x}"
    )


def _optimize(root: Op) -> Op:
    return PassManager((StructureDiscoveryPass(),)).run(root)


@dataclass(frozen=True, slots=True)
class _Repeat(Op):
    body: Op
    count: int

    @property
    def operands(self) -> tuple[Op, ...]:
        return (self.body,)

    def with_operands(self, operands: tuple[Op, ...]) -> Op:
        if len(operands) != 1:
            raise ValueError("repeat requires one operand")
        return _Repeat(operands[0], self.count)


class _RepeatLowerer(OpLowerer[str]):
    @property
    def op_type(self) -> type[Op]:
        return _Repeat

    def lower(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        assert isinstance(op, _Repeat)
        return operands[0] * op.count


class _FoldZeroWidthConcat(RewritePattern):
    root_type = Concat

    def match_and_rewrite(self, op: Op, rewriter: PatternRewriter) -> bool:
        properties = rewriter.get_analysis(RegexPropertiesAnalysis)
        if (
            properties.can_match
            and properties.min_width == 0
            and properties.max_width == 0
        ):
            rewriter.replace_op(op, EPSILON)
            return True
        return False


class REIRInfrastructureTests(unittest.TestCase):
    def test_extension_points_are_abstract(self) -> None:
        for extension_point in (
            Op,
            PureOp,
            DataFlowAnalysis,
            RewritePattern,
            OperationPass,
            Lowerer,
            OpLowerer,
            CostModel,
            CandidateGenerator,
            CandidateSelector,
        ):
            with self.subTest(extension_point=extension_point.__name__):
                self.assertTrue(inspect.isabstract(extension_point))

    def test_core_analysis_propagates_language_and_cost_facts(self) -> None:
        expression = concat(
            literal(b"ab"),
            repeat(charset((ord("c"), ord("d"))), 0, 2),
        )
        properties = AnalysisManager().get(RegexPropertiesAnalysis, expression)
        self.assertTrue(properties.can_match)
        self.assertFalse(properties.nullable)
        self.assertEqual((properties.min_width, properties.max_width), (2, 4))
        self.assertEqual(properties.first_symbols, frozenset((ord("a"),)))
        self.assertEqual(
            properties.last_symbols,
            frozenset((ord("b"), ord("c"), ord("d"))),
        )
        self.assertEqual(properties.operation_count, 4)
        self.assertEqual(properties.literal_bytes, 2)

    def test_pattern_can_consume_analysis_facts_through_rewriter(self) -> None:
        raw = Concat((EPSILON, EPSILON))
        rewritten = GreedyRewriteDriver((_FoldZeroWidthConcat(),)).rewrite(raw)
        self.assertIs(rewritten, EPSILON)

    def test_custom_op_and_lowering_rule_extend_the_source_compiler(self) -> None:
        lowerer = RegexSourceLowerer(escape_byte=_ascii_escape)
        lowerer.register(_RepeatLowerer())
        compiler = RegexCompiler(lowerer)
        self.assertEqual(compiler.compile(_Repeat(literal(b"a"), 3)), "aaa")

    def test_canonicalization_is_an_explicit_compiler_pass(self) -> None:
        raw = Concat((literal(b"a"), Concat((EPSILON, literal(b"b")))))
        compiler = RegexCompiler(
            RegexSourceLowerer(escape_byte=_ascii_escape),
            passes=(CanonicalizePass(),),
        )
        result = compiler.run(raw)
        self.assertEqual(result.ir, literal(b"ab"))
        self.assertEqual(result.output, "ab")


class REIRCostAndBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lowerer = RegexSourceLowerer(escape_byte=_ascii_escape)

    def test_structural_and_lowered_cost_models_share_stable_tie_breakers(
        self,
    ) -> None:
        expression = concat(literal(b"ab"), repeat(literal(b"c"), 0, 1))
        structural = StructuralCostModel().evaluate(expression)
        self.assertEqual(structural, StructuralCost(4, 3))

        source = SourceSizeCostModel(self.lowerer).evaluate(expression)
        self.assertEqual(source, LoweredSizeCost(len("abc?"), 4, 3))

        deflated = DeflatedSourceCostModel(self.lowerer).evaluate(expression)
        self.assertEqual(
            deflated,
            LoweredSizeCost(raw_deflate_size("abc?"), 4, 3),
        )

    def test_functional_cost_model_can_measure_complete_external_context(self) -> None:
        model = FunctionalCostModel(
            lambda root, analyses: (
                analyses.get(RegexPropertiesAnalysis, root).operation_count,
                render_regex(root, escape_byte=_ascii_escape),
            ),
            key=lambda cost: (cost[0], len(cost[1])),
        )
        self.assertEqual(model.evaluate(literal(b"abc")), (1, "abc"))

    def test_minimum_selector_uses_cost_and_preserves_the_first_tie(self) -> None:
        original = alternate(literal(b"fooa"), literal(b"foob"))
        factored = concat(literal(b"foo"), charset(b"ab"))
        selector = MinimumCostSelector()
        model = SourceSizeCostModel(self.lowerer)

        selected = selector.select((original, factored), model)
        self.assertEqual(selected.root, factored)
        self.assertEqual(selected.ordinal, 1)
        self.assertEqual(selected.cost.size, len("foo[ab]"))

        tied = selector.select((literal(b"a"), literal(b"b")), model)
        self.assertEqual(tied.root, literal(b"a"))
        self.assertEqual(tied.ordinal, 0)

    def test_minimum_selector_rejects_an_empty_candidate_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            MinimumCostSelector().select((), StructuralCostModel())

    def test_candidate_selection_pass_connects_generator_cost_and_compiler(
        self,
    ) -> None:
        original = alternate(literal(b"fooa"), literal(b"foob"))
        factored = concat(literal(b"foo"), charset(b"ab"))
        generator = FunctionalCandidateGenerator(lambda root, analyses: (factored,))
        compiler = RegexCompiler(
            self.lowerer,
            passes=(
                CandidateSelectionPass(
                    (generator,),
                    SourceSizeCostModel(self.lowerer),
                ),
            ),
        )
        result = compiler.run(original)
        self.assertEqual(result.ir, factored)
        self.assertEqual(result.output, "foo[ab]")

    def test_lowered_cost_rejects_a_negative_measurement(self) -> None:
        model = LoweredSizeCostModel(self.lowerer, lambda source: -len(source))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            model.evaluate(literal(b"abc"))

    def test_compiler_benchmark_reports_optimized_ir_source_and_timing(self) -> None:
        expression = alternate(literal(b"foo"), literal(b"foobar"))
        compiler = RegexCompiler(
            self.lowerer,
            passes=(StructureDiscoveryPass(),),
        )
        timestamps = iter((10.0, 12.0))
        result = benchmark_compiler(
            expression,
            compiler,
            iterations=2,
            clock=lambda: next(timestamps),
        )
        self.assertEqual(result.compilation.output, "foo(?:bar)?")
        self.assertEqual(result.metrics.iterations, 2)
        self.assertEqual(result.metrics.seconds_per_iteration, 1.0)
        self.assertEqual(result.metrics.operation_count, 4)
        self.assertEqual(result.metrics.literal_bytes, 6)
        self.assertEqual(result.metrics.source_characters, len("foo(?:bar)?"))
        self.assertEqual(result.metrics.source_bytes, len(b"foo(?:bar)?"))
        self.assertEqual(
            result.metrics.deflated_source_bytes,
            raw_deflate_size("foo(?:bar)?"),
        )

    def test_compiler_benchmark_requires_a_positive_iteration_count(self) -> None:
        compiler = RegexCompiler(self.lowerer)
        with self.assertRaisesRegex(ValueError, "positive"):
            benchmark_compiler(literal(b"a"), compiler, iterations=0)


class PureRegexCanonicalizationTests(unittest.TestCase):
    def test_charset_is_a_canonical_byte_bitset(self) -> None:
        expression = charset(b"dcabx")
        self.assertIsInstance(expression, CharSet)
        assert isinstance(expression, CharSet)
        self.assertEqual(expression.symbols, frozenset(b"abcdx"))
        self.assertEqual(
            expression.intervals, ((ord("a"), ord("d")), (ord("x"), ord("x")))
        )
        self.assertIs(charset(()), NEVER)
        complement = expression.complement()
        self.assertIsInstance(complement, CharSet)
        assert isinstance(complement, CharSet)
        self.assertNotIn(ord("a"), complement.symbols)

    def test_alternate_is_flat_sorted_unique_and_combines_charsets(self) -> None:
        expression = alternate(
            literal(b"b"),
            NEVER,
            alternate(literal(b"a"), literal(b"b")),
            charset((ord("c"), ord("d"))),
        )
        self.assertEqual(expression, CharSet(b"abcd"))

        ordered = alternate(literal(b"zz"), literal(b"aa"), literal(b"zz"))
        self.assertEqual(ordered, Alternate((Literal(b"aa"), Literal(b"zz"))))

    def test_concat_folds_identities_literals_and_adjacent_repeats(self) -> None:
        body = literal(b"a")
        self.assertEqual(
            concat(EPSILON, literal(b"foo"), literal(b"bar")), literal(b"foobar")
        )
        self.assertIs(concat(body, NEVER), NEVER)
        self.assertEqual(
            concat(body, repeat(body, 0, None)),
            Repeat(body, 1, None),
        )
        self.assertEqual(
            concat(repeat(body, 0, None), repeat(body, 0, None)),
            Repeat(body, 0, None),
        )

    def test_repeat_folds_trivial_and_closure_cases(self) -> None:
        body = literal(b"a")
        self.assertEqual(repeat(body, 1, 1), body)
        self.assertIs(repeat(body, 0, 0), EPSILON)
        self.assertIs(repeat(EPSILON, 3, None), EPSILON)
        self.assertIs(repeat(NEVER, 0, None), EPSILON)
        self.assertIs(repeat(NEVER, 1, None), NEVER)
        star = repeat(body, 0, None)
        self.assertEqual(repeat(star, 0, None), star)

    def test_invalid_charset_and_repeat_bounds_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "byte alphabet"):
            CharSet((256,))
        with self.assertRaisesRegex(ValueError, "minimum"):
            Repeat(literal(b"a"), -1, None)
        with self.assertRaisesRegex(ValueError, "maximum"):
            Repeat(literal(b"a"), 2, 1)


class PureRegexStructureDiscoveryTests(unittest.TestCase):
    def test_longest_common_prefix_is_factored(self) -> None:
        optimized = _optimize(alternate(literal(b"foo"), literal(b"foobar")))
        self.assertEqual(
            optimized,
            concat(literal(b"foo"), repeat(literal(b"bar"), 0, 1)),
        )

    def test_longest_common_suffix_is_factored(self) -> None:
        optimized = _optimize(alternate(literal(b"bar"), literal(b"foobar")))
        self.assertEqual(
            optimized,
            concat(repeat(literal(b"foo"), 0, 1), literal(b"bar")),
        )

    def test_literal_power_union_is_discovered(self) -> None:
        optimized = _optimize(
            alternate(
                literal(b"a"),
                literal(b"aa"),
                literal(b"aaa"),
                literal(b"aaaa"),
            )
        )
        self.assertEqual(optimized, Repeat(literal(b"a"), 1, 4))

    def test_common_context_and_expression_powers_are_discovered(self) -> None:
        prefix = literal(b"x")
        body = charset((ord("a"), ord("b")))
        suffix = literal(b"!")
        optimized = _optimize(
            alternate(
                concat(prefix, body, suffix),
                concat(prefix, repeat(body, 2, 2), suffix),
                concat(prefix, repeat(body, 3, 3), suffix),
            )
        )
        self.assertEqual(
            optimized,
            concat(prefix, Repeat(body, 1, 3), suffix),
        )

    def test_repeat_absorbs_a_contained_alternative(self) -> None:
        body = literal(b"a")
        self.assertEqual(
            _optimize(alternate(body, repeat(body, 0, None))),
            Repeat(body, 0, None),
        )

    def test_renderer_selects_charset_ranges_and_quantifiers(self) -> None:
        self.assertEqual(
            render_regex(
                alternate(*(literal(bytes((symbol,))) for symbol in b"abcd")),
                escape_byte=_ascii_escape,
            ),
            "[a-d]",
        )
        self.assertEqual(
            render_regex(
                repeat(literal(b"ab"), 2, 3),
                escape_byte=_ascii_escape,
            ),
            "(?:ab){2,3}",
        )

    def test_renderer_selects_derivative_factoring_only_after_deflate_wins(
        self,
    ) -> None:
        tail = literal(b"a-long-common-residual-with-locality-0123456789")
        expression = alternate(
            *(concat(literal(bytes((head,))), tail) for head in b"abde"),
            literal(b"cx"),
        )
        self.assertEqual(
            render_regex(expression, escape_byte=_ascii_escape),
            "(?:cx|[abde]a\\x2dlong\\x2dcommon\\x2dresidual"
            "\\x2dwith\\x2dlocality\\x2d0123456789)",
        )

    def test_optimizer_preserves_random_small_languages_exhaustively(self) -> None:
        randomizer = random.Random(20_260_821)
        leaves = (
            EPSILON,
            NEVER,
            literal(b"a"),
            literal(b"b"),
            literal(b"c"),
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
                ((0, 1), (0, 2), (0, None), (1, 3), (2, 2))
            )
            return repeat(generate(depth - 1), minimum, maximum)

        candidates = tuple(
            bytes(values)
            for length in range(5)
            for values in itertools.product(b"abc", repeat=length)
        )
        raw_compiler = RegexCompiler(RegexSourceLowerer(escape_byte=_ascii_escape))
        for _ in range(100):
            expression = generate(3)
            optimized = _optimize(expression)
            before = re.compile(raw_compiler.compile(expression).encode("ascii"))
            after = re.compile(raw_compiler.compile(optimized).encode("ascii"))
            for candidate in candidates:
                self.assertEqual(
                    before.fullmatch(candidate) is not None,
                    after.fullmatch(candidate) is not None,
                    (expression, optimized, candidate),
                )


class TaggedRegexSafetyTests(unittest.TestCase):
    def test_pure_builder_rejects_tagged_operands(self) -> None:
        with self.assertRaisesRegex(TypeError, "pure regex"):
            concat(literal(b"a"), tagged(1))

    def test_tagged_alternate_preserves_order_and_duplicates(self) -> None:
        expression = TaggedAlternate((tagged(2), tagged(1), tagged(2)))
        rewritten = PassManager((StructureDiscoveryPass(),)).run(expression)
        self.assertEqual(rewritten, expression)

    def test_repeat_rejects_tagged_output_semantics(self) -> None:
        with self.assertRaisesRegex(TypeError, "not defined"):
            TAGGED_BUILDER.repeat(tagged(1), 0, None)

    def test_lowering_visits_each_syntactic_tag_occurrence_in_a_dag(self) -> None:
        ranks: list[int] = []
        shared_tag = tagged(7)
        expression = TaggedConcat((shared_tag, shared_tag))
        source = render_tagged_regex(
            expression,
            escape_byte=_ascii_escape,
            emit_tag=lambda rank: ranks.append(rank) or "()",
        )
        self.assertEqual(source, "()()")
        self.assertEqual(ranks, [7, 7])


if __name__ == "__main__":
    unittest.main()
