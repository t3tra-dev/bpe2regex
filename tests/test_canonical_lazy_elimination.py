import itertools
import unittest

from bpe2regex import BoundaryRegexBPE
from bpe2regex.reir import (
    CanonicalLazyBoundaryRegexCompiler,
    LazyEliminationBudget,
    LazyEliminationBudgetExceeded,
)
from bpe2regex.vocabulary import reference_bpe_ids
from tests.test_canonical_token_dfa import PARENTS, RANK_OF, TOKENS


class CanonicalLazyBoundaryCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = CanonicalLazyBoundaryRegexCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        cls.compiled = cls.compiler.compile_python()

    def test_lazy_pipeline_emits_reference_tokenization(self) -> None:
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

    def test_pipeline_reports_each_lazy_stage(self) -> None:
        metrics = self.compiled.ir.metrics
        self.assertEqual(metrics.adjacency.applied_merges, len(TOKENS) - 3)
        self.assertLessEqual(
            metrics.quotient.quotient_state_count,
            metrics.quotient.reachable_state_count,
        )
        self.assertEqual(
            metrics.elimination.materialized_row_count,
            metrics.quotient.quotient_state_count,
        )
        self.assertEqual(
            metrics.elimination.eliminated_state_count,
            metrics.quotient.quotient_state_count,
        )

    def test_elimination_budget_stops_before_row_materialization(self) -> None:
        with self.assertRaises(LazyEliminationBudgetExceeded) as raised:
            self.compiler.compile_python(
                budget=LazyEliminationBudget(max_states=1),
            )
        self.assertEqual(raised.exception.reason, "quotient state count")
        self.assertEqual(raised.exception.metrics.materialized_row_count, 0)

    def test_merge_prefix_uses_only_active_token_lookup_entries(self) -> None:
        compiled = self.compiler.compile_python(merge_limit=3)
        self.assertEqual(set(compiled.token_capture_ranks), set(range(6)))
        with BoundaryRegexBPE(compiled) as program:
            word = b"aaabcb"
            match = program.fullmatch(word)
            self.assertIsNotNone(match)
            assert match is not None
            self.assertEqual(
                match.token_ids,
                reference_bpe_ids(word, TOKENS, RANK_OF, cutoff=6),
            )


if __name__ == "__main__":
    unittest.main()
