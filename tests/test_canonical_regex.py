import itertools
import re
import unittest

from bpe2regex import CanonicalRegexBPE
from bpe2regex.reir import (
    NEVER,
    BooleanTokenSymbolLowerer,
    CanonicalTokenRegexCompiler,
    Difference,
    Op,
    SymbolSet,
    TokenSymbolLowerer,
    lower_boolean_to_core,
    raw_deflate_size,
)
from bpe2regex.reir.source import render_regex
from bpe2regex.vocabulary import recover_merge_parents, reference_bpe_ids
from tests.test_canonical_token_dfa import PARENTS, RANK_OF, TOKENS


def _python_pattern(expression: Op) -> str:
    return render_regex(
        expression,
        escape_byte=lambda byte: f"\\x{byte:02x}",
    )


class TokenSymbolLowererTests(unittest.TestCase):
    def test_token_sets_lower_to_the_same_finite_byte_language(self) -> None:
        lowerer = TokenSymbolLowerer(TOKENS)
        selected = (0, 3, 6, 8, 9)
        expression = lowerer(SymbolSet.from_symbols(len(TOKENS), selected))
        pattern = re.compile(_python_pattern(expression).encode("ascii"))
        for rank, token in enumerate(TOKENS):
            self.assertEqual(pattern.fullmatch(token) is not None, rank in selected)
        self.assertIs(
            lowerer(SymbolSet.empty(len(TOKENS))),
            NEVER,
        )

    def test_dense_token_sets_use_a_finite_language_difference(self) -> None:
        domain = SymbolSet.full(len(TOKENS))
        lowerer = BooleanTokenSymbolLowerer(TOKENS, domain)
        selected = tuple(range(1, len(TOKENS)))
        expression = lowerer(SymbolSet.from_symbols(len(TOKENS), selected))
        self.assertIsInstance(expression, Difference)

        pattern = re.compile(
            _python_pattern(lower_boolean_to_core(expression)).encode("ascii")
        )
        for rank, token in enumerate(TOKENS):
            self.assertEqual(pattern.fullmatch(token) is not None, rank in selected)


class CanonicalTokenRegexCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compilation = CanonicalTokenRegexCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        ).compile_python()

    def test_repeated_fullmatches_emit_every_canonical_boundary(self) -> None:
        program = CanonicalRegexBPE(self.compilation)
        try:
            for length in range(7):
                for values in itertools.product(b"abc", repeat=length):
                    word = bytes(values)
                    expected = reference_bpe_ids(word, TOKENS, RANK_OF)
                    match = program.fullmatch(word)
                    self.assertIsNotNone(match, word)
                    assert match is not None
                    self.assertEqual(match.token_ids, expected, word)
                    self.assertEqual(b"".join(match.captures()), word)
        finally:
            program.close()

    def test_minimized_and_unminimized_automata_emit_the_same_tokens(self) -> None:
        unminimized = CanonicalTokenRegexCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        ).compile_python(minimize=False)
        programs = (CanonicalRegexBPE(self.compilation), CanonicalRegexBPE(unminimized))
        try:
            for word in (b"", b"abc", b"aaab", b"bcbcaab", b"aaaaaa"):
                expected = reference_bpe_ids(word, TOKENS, RANK_OF)
                for program in programs:
                    match = program.fullmatch(word)
                    self.assertIsNotNone(match)
                    assert match is not None
                    self.assertEqual(match.token_ids, expected)
        finally:
            for program in programs:
                program.close()

    def test_bytes_outside_the_base_alphabet_do_not_match(self) -> None:
        with CanonicalRegexBPE(self.compilation) as program:
            self.assertIsNone(program.fullmatch(b"d"))

    def test_capture_ranks_cover_the_dictionary(self) -> None:
        self.assertEqual(
            set(self.compilation.capture_ranks),
            set(range(len(TOKENS))),
        )

    def test_cost_search_never_loses_the_fixed_scc_order(self) -> None:
        searched = CanonicalTokenRegexCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        ).compile_python(elimination_beam_width=3)
        self.assertLessEqual(
            raw_deflate_size(searched.pattern),
            raw_deflate_size(self.compilation.pattern),
        )
        self.assertGreater(
            searched.ir.metrics.explored_elimination_candidates,
            0,
        )
        with CanonicalRegexBPE(searched) as program:
            for length in range(5):
                for values in itertools.product(b"abc", repeat=length):
                    word = bytes(values)
                    match = program.fullmatch(word)
                    self.assertIsNotNone(match)
                    assert match is not None
                    self.assertEqual(
                        match.token_ids,
                        reference_bpe_ids(word, TOKENS, RANK_OF),
                    )

    def test_cost_search_rejects_an_empty_beam(self) -> None:
        compiler = CanonicalTokenRegexCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            compiler.compile_ir(elimination_beam_width=0)

    def test_boolean_token_labels_lower_before_source_emission(self) -> None:
        compiled = CanonicalTokenRegexCompiler(
            TOKENS,
            PARENTS,
            base_token_count=3,
        ).compile_python(merge_limit=1, boolean_token_labels=True)
        self.assertIsNotNone(compiled.ir.symbolic_expression)
        self.assertGreater(compiled.ir.metrics.boolean_lowering_seconds, 0.0)
        with CanonicalRegexBPE(compiled) as program:
            for length in range(6):
                for values in itertools.product(b"abc", repeat=length):
                    word = bytes(values)
                    match = program.fullmatch(word)
                    self.assertIsNotNone(match)
                    assert match is not None
                    self.assertEqual(
                        match.token_ids,
                        reference_bpe_ids(word, TOKENS, RANK_OF, cutoff=4),
                    )


class CanonicalRegexValidationTests(unittest.TestCase):
    def test_duplicate_base_tokens_are_rejected(self) -> None:
        tokens = (b"a", b"a")
        parents = recover_merge_parents(tokens, {b"a": 0}, base_token_count=2)
        with self.assertRaisesRegex(ValueError, "unique"):
            CanonicalTokenRegexCompiler(tokens, parents, base_token_count=2)


if __name__ == "__main__":
    unittest.main()
