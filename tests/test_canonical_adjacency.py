import unittest

from bpe2regex.reir import CanonicalAdjacencyCompiler, CanonicalTokenDFACompiler
from tests.test_canonical_token_dfa import PARENTS, TOKENS


class CanonicalAdjacencyCompilerTests(unittest.TestCase):
    def test_persistent_cells_equal_the_materialized_canonical_dfa(self) -> None:
        persistent_compiler = CanonicalAdjacencyCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        dense_compiler = CanonicalTokenDFACompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        for merge_limit in (0, 1, 3, None):
            with self.subTest(merge_limit=merge_limit):
                persistent = persistent_compiler.compile(
                    merge_limit=merge_limit
                ).adjacency
                dense = dense_compiler.compile(merge_limit=merge_limit).automaton
                self.assertEqual(persistent.state_count, dense.state_count)
                for state in range(dense.state_count):
                    for token in range(len(TOKENS)):
                        self.assertEqual(
                            persistent.transition(state, token),
                            dense.transition(state, token),
                            (state, token),
                        )

    def test_materialized_persistent_dfa_is_identical(self) -> None:
        result = CanonicalAdjacencyCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        ).compile()
        dense = (
            CanonicalTokenDFACompiler(
                TOKENS,
                PARENTS,
                base_token_count=3,
            )
            .compile()
            .automaton
        )
        self.assertEqual(result.adjacency.to_dfa(), dense)

    def test_storage_is_linear_while_logical_cells_are_quadratic(self) -> None:
        result = CanonicalAdjacencyCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        ).compile()
        metrics = result.metrics
        self.assertEqual(metrics.state_count, metrics.applied_merges + 1)
        self.assertEqual(metrics.token_parent_links, metrics.applied_merges)
        self.assertEqual(metrics.state_parent_links, metrics.applied_merges)
        self.assertLess(metrics.persistent_record_count, metrics.dense_cell_count)

    def test_large_materialization_requires_an_explicit_budget(self) -> None:
        adjacency = (
            CanonicalAdjacencyCompiler(
                TOKENS,
                PARENTS,
                base_token_count=3,
            )
            .compile()
            .adjacency
        )
        with self.assertRaisesRegex(RuntimeError, "budget"):
            adjacency.to_dfa(max_cells=1)

    def test_inactive_tokens_are_denied_without_clone_traversal(self) -> None:
        adjacency = (
            CanonicalAdjacencyCompiler(
                TOKENS,
                PARENTS,
                base_token_count=3,
            )
            .compile(merge_limit=0)
            .adjacency
        )
        for token in range(3, len(TOKENS)):
            self.assertFalse(adjacency.allowed(0, token))
            self.assertIsNone(adjacency.transition(0, token))


if __name__ == "__main__":
    unittest.main()
