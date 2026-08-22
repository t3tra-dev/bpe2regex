import itertools
import re
import unittest

from bpe2regex import BoundaryRegexBPE
from bpe2regex.reir import (
    BOUNDARY,
    DEFAULT_BUILDER,
    AnalysisManager,
    BoundaryCostObjective,
    CanonicalBoundaryRegexCompiler,
    MarkerCountAnalysis,
    render_marked_regex,
    verify_single_boundary,
)
from bpe2regex.vocabulary import reference_bpe_ids
from tests.test_canonical_token_dfa import PARENTS, RANK_OF, TOKENS


class MarkedRegexIRTests(unittest.TestCase):
    def test_marker_count_proves_exactly_one_boundary(self) -> None:
        expression = DEFAULT_BUILDER.alternate(
            DEFAULT_BUILDER.concat(DEFAULT_BUILDER.literal(b"a"), BOUNDARY),
            DEFAULT_BUILDER.concat(BOUNDARY, DEFAULT_BUILDER.literal(b"b")),
        )
        count = AnalysisManager().get(MarkerCountAnalysis, expression)
        self.assertTrue(count.is_exactly_one)
        verify_single_boundary(expression)

    def test_marker_count_rejects_optional_or_repeated_boundaries(self) -> None:
        for expression in (
            DEFAULT_BUILDER.repeat(BOUNDARY, 0, 1),
            DEFAULT_BUILDER.repeat(BOUNDARY, 1, None),
            DEFAULT_BUILDER.alternate(BOUNDARY, DEFAULT_BUILDER.literal(b"x")),
        ):
            with (
                self.subTest(expression=expression),
                self.assertRaisesRegex(ValueError, "exactly one"),
            ):
                verify_single_boundary(expression)

    def test_marked_lowering_erases_boundary_to_one_capture(self) -> None:
        expression = DEFAULT_BUILDER.concat(
            DEFAULT_BUILDER.literal(b"ab"),
            BOUNDARY,
            DEFAULT_BUILDER.literal(b"c"),
        )
        source = render_marked_regex(
            expression,
            escape_byte=lambda byte: f"\\x{byte:02x}",
            emit_boundary=lambda: "()",
        )
        match = re.fullmatch(source.encode("ascii"), b"abc")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.start(1), 2)


class CanonicalBoundaryRegexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiled = CanonicalBoundaryRegexCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        ).compile_python(elimination_beam_width=3)

    def test_repeated_matches_and_lookup_emit_reference_tokens(self) -> None:
        with BoundaryRegexBPE(self.compiled) as program:
            for length in range(7):
                for values in itertools.product(b"abc", repeat=length):
                    word = bytes(values)
                    match = program.fullmatch(word)
                    self.assertIsNotNone(match, word)
                    assert match is not None
                    self.assertEqual(
                        match.token_ids,
                        reference_bpe_ids(word, TOKENS, RANK_OF),
                        word,
                    )
                    self.assertEqual(b"".join(match.captures()), word)

    def test_every_fullmatch_selects_one_boundary_capture(self) -> None:
        pattern = re.compile(self.compiled.boundary_pattern.encode("ascii"))
        for length in range(1, 6):
            for values in itertools.product(b"abc", repeat=length):
                match = pattern.fullmatch(bytes(values))
                self.assertIsNotNone(match)
                assert match is not None
                participating = sum(
                    match.start(group) >= 0 for group in range(1, pattern.groups + 1)
                )
                self.assertEqual(participating, 1)

    def test_full_artifact_cost_includes_both_patterns_and_rank_table(self) -> None:
        cost = self.compiled.cost
        source_bytes = len(
            (self.compiled.boundary_pattern + self.compiled.token_to_rank).encode(
                "ascii"
            )
        )
        self.assertEqual(cost.source_bytes, source_bytes)
        self.assertGreater(cost.artifact_bytes, 0)
        self.assertGreater(len(self.compiled.token_capture_ranks), 0)

    def test_all_cost_objectives_produce_equivalent_programs(self) -> None:
        compiler = CanonicalBoundaryRegexCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        for objective in BoundaryCostObjective:
            with self.subTest(objective=objective):
                compiled = compiler.compile_python(
                    merge_limit=3,
                    elimination_beam_width=2,
                    objective=objective,
                )
                with BoundaryRegexBPE(compiled) as program:
                    for word in (b"", b"abc", b"aaab", b"bcbcaab"):
                        match = program.fullmatch(word)
                        self.assertIsNotNone(match)
                        assert match is not None
                        self.assertEqual(
                            match.token_ids,
                            reference_bpe_ids(word, TOKENS, RANK_OF, cutoff=6),
                        )


if __name__ == "__main__":
    unittest.main()
