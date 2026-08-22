import unittest

from bpe2regex.reir import (
    CanonicalAdjacencyCompiler,
    CanonicalLazyQuotientCompiler,
    CanonicalTokenDFACompiler,
    equivalence_counterexample,
    minimize_dfa,
    prune_dead_states,
)
from tests.test_canonical_token_dfa import PARENTS, TOKENS


class CanonicalLazyQuotientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.persistent_compiler = CanonicalAdjacencyCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        self.dense_compiler = CanonicalTokenDFACompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )

    def test_denied_signatures_equal_every_reachable_persistent_row(self) -> None:
        adjacency = self.persistent_compiler.compile().adjacency
        quotient = CanonicalLazyQuotientCompiler().compile(adjacency)
        for raw_state, quotient_state in enumerate(quotient.state_map):
            if quotient_state is None:
                continue
            signature = quotient.signatures[quotient_state]
            for token in adjacency.active_tokens:
                self.assertEqual(
                    quotient.clone_index.denies(signature, token),
                    not adjacency.allowed(raw_state, token),
                    (raw_state, token),
                )

    def test_lazy_quotient_matches_dense_minimization(self) -> None:
        for merge_limit in (0, 1, 3, None):
            with self.subTest(merge_limit=merge_limit):
                adjacency = self.persistent_compiler.compile(
                    merge_limit=merge_limit
                ).adjacency
                quotient = CanonicalLazyQuotientCompiler().compile(adjacency)
                lazy_dfa = quotient.to_dfa()
                dense = self.dense_compiler.compile(merge_limit=merge_limit).automaton
                minimized = prune_dead_states(minimize_dfa(dense).automaton).automaton
                self.assertEqual(quotient.state_count, minimized.state_count)
                self.assertIsNone(equivalence_counterexample(lazy_dfa, minimized))

    def test_quotient_rows_are_materialized_on_demand(self) -> None:
        adjacency = self.persistent_compiler.compile().adjacency
        quotient = CanonicalLazyQuotientCompiler().compile(adjacency)
        self.assertEqual(quotient._row_cache, {})
        quotient.transition_groups(quotient.start)
        self.assertEqual(tuple(quotient._row_cache), (quotient.start,))

    def test_materialization_budget_uses_quotient_cells(self) -> None:
        adjacency = self.persistent_compiler.compile().adjacency
        quotient = CanonicalLazyQuotientCompiler().compile(adjacency)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            quotient.to_dfa(max_cells=1)


if __name__ == "__main__":
    unittest.main()
