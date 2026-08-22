import itertools
import unittest

from bpe2regex.reir import (
    CanonicalTokenDFABudgetExceeded,
    CanonicalTokenDFACompiler,
)
from bpe2regex.vocabulary import recover_merge_parents, reference_bpe_ids

TOKENS = (
    b"a",
    b"b",
    b"c",
    b"aa",
    b"bc",
    b"ab",
    b"aab",
    b"bcbc",
    b"abc",
    b"aaab",
)
RANK_OF = {token: rank for rank, token in enumerate(TOKENS)}
PARENTS = recover_merge_parents(TOKENS, RANK_OF, base_token_count=3)


def _tokenizations(value: bytes) -> tuple[tuple[int, ...], ...]:
    results: list[tuple[int, ...]] = []

    def visit(position: int, ranks: tuple[int, ...]) -> None:
        if position == len(value):
            results.append(ranks)
            return
        for rank, token in enumerate(TOKENS):
            if value.startswith(token, position):
                visit(position + len(token), (*ranks, rank))

    visit(0, ())
    return tuple(results)


class CanonicalTokenDFACompilerTests(unittest.TestCase):
    def test_accepts_exactly_the_canonical_tokenization_of_every_small_word(
        self,
    ) -> None:
        result = CanonicalTokenDFACompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        ).compile()
        automaton = result.automaton
        self.assertEqual(result.metrics.applied_merges, 7)
        self.assertTrue(automaton.accepts(()))

        for length in range(7):
            for values in itertools.product(b"abc", repeat=length):
                word = bytes(values)
                expected = tuple(reference_bpe_ids(word, TOKENS, RANK_OF))
                candidates = _tokenizations(word)
                self.assertIn(expected, candidates)
                for candidate in candidates:
                    self.assertEqual(
                        automaton.accepts(candidate),
                        candidate == expected,
                        (word, candidate, expected),
                    )

    def test_merge_prefixes_build_the_corresponding_partial_dictionary(self) -> None:
        compiler = CanonicalTokenDFACompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        for merge_limit in range(8):
            automaton = compiler.compile(merge_limit=merge_limit).automaton
            cutoff = 3 + merge_limit
            for length in range(5):
                for values in itertools.product(b"abc", repeat=length):
                    word = bytes(values)
                    expected = reference_bpe_ids(
                        word,
                        TOKENS,
                        RANK_OF,
                        cutoff=cutoff,
                    )
                    self.assertTrue(automaton.accepts(expected))

    def test_state_budget_reports_the_last_complete_merge(self) -> None:
        compiler = CanonicalTokenDFACompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        with self.assertRaises(CanonicalTokenDFABudgetExceeded) as raised:
            compiler.compile(max_states=1)
        self.assertEqual(raised.exception.reason, "state count")
        self.assertEqual(raised.exception.metrics.applied_merges, 0)
        self.assertEqual(raised.exception.metrics.state_count, 1)

    def test_invalid_merge_limit_is_rejected(self) -> None:
        compiler = CanonicalTokenDFACompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compiler.compile(merge_limit=-1)


if __name__ == "__main__":
    unittest.main()
