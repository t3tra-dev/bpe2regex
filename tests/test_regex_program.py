import itertools
import unittest

from bpe2regex.binary import (
    decode_ecmascript_artifact,
    decode_python_artifact,
    encode_artifact,
)
from bpe2regex.emitter import Compatibility, emit_regex_sources
from bpe2regex.emitter.ecmascript import (
    RegexSources as ECMAScriptRegexSources,
)
from bpe2regex.emitter.ecmascript import (
    validate_sources as validate_ecmascript_sources,
)
from bpe2regex.emitter.python import RegexSources as PythonRegexSources
from bpe2regex.encoding import Encoding
from bpe2regex.regex_program import RegexBPE
from bpe2regex.vocabulary import recover_merge_parents, reference_bpe_ids


def toy_program() -> tuple[RegexBPE, tuple[bytes, ...], dict[bytes, int]]:
    tokens = (
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
    rank_of = {token: rank for rank, token in enumerate(tokens)}
    parents = recover_merge_parents(tokens, rank_of, base_token_count=3)
    sources = emit_regex_sources(
        tokens,
        parents,
        Compatibility.PYTHON,
        base_token_count=3,
    )
    return RegexBPE(sources), tokens, rank_of


class StdlibRegexProgramTests(unittest.TestCase):
    def test_matches_reference_exhaustively(self) -> None:
        program, tokens, rank_of = toy_program()
        with program:
            for length in range(8):
                for values in itertools.product(b"abc", repeat=length):
                    piece = bytes(values)
                    match = program.fullmatch(piece)
                    self.assertIsNotNone(match)
                    assert match is not None
                    expected = reference_bpe_ids(piece, tokens, rank_of)
                    self.assertEqual(match.token_ids, expected, piece)
                    self.assertEqual(b"".join(match.captures()), piece)
                    self.assertEqual(
                        [piece[start:end] for start, end in match.spans()],
                        match.captures(),
                    )

    def test_higher_priority_overlapping_merge_wins(self) -> None:
        tokens = (b"a", b"b", b"c", b"bc", b"ab")
        rank_of = {token: rank for rank, token in enumerate(tokens)}
        parents = recover_merge_parents(tokens, rank_of, base_token_count=3)
        program = RegexBPE(
            emit_regex_sources(
                tokens,
                parents,
                Compatibility.PYTHON,
                base_token_count=3,
            )
        )
        match = program.fullmatch(b"abc")
        assert match is not None
        self.assertEqual(match.token_ids, [0, 3])
        self.assertEqual(match.captures(), [b"a", b"bc"])

    def test_patterns_embed_base_and_merge_ranks(self) -> None:
        program, _, _ = toy_program()
        self.assertEqual(set(program.byte_pattern.groupindex), {"b0", "b1", "b2"})
        self.assertEqual(
            set(program.merge_pattern.groupindex),
            {"m3", "m4", "m5", "m6", "m7", "m8", "m9"},
        )
        self.assertFalse(program.sources.merge_pair.startswith("(?="))

    def test_python_binary_round_trip_and_compression(self) -> None:
        program, _, _ = toy_program()
        encoded = encode_artifact(
            Encoding.R50K,
            Compatibility.PYTHON,
            program.sources,
            r"[a-z]+|.",
        )
        encoding, sources, pretokenizer = decode_python_artifact(encoded)
        self.assertIs(encoding, Encoding.R50K)
        self.assertEqual(sources, program.sources)
        self.assertEqual(pretokenizer, r"[a-z]+|.")
        self.assertNotIn(b"(?P<", encoded)

        damaged = bytes((encoded[0] ^ 0xFF,)) + encoded[1:]
        with self.assertRaisesRegex(ValueError, "decompress"):
            decode_python_artifact(damaged)

    def test_missing_rank_group_is_rejected(self) -> None:
        program, _, _ = toy_program()
        damaged = program.sources.byte_to_rank.replace("(?P<b0>", "(?:", 1)
        sources = PythonRegexSources(
            byte_to_rank=damaged,
            merge_pair=program.sources.merge_pair,
            token_count=program.sources.token_count,
            base_token_count=program.sources.base_token_count,
            rank_width=program.sources.rank_width,
        )
        with self.assertRaisesRegex(ValueError, "rank capture groups differ"):
            RegexBPE(sources)

    def test_ecmascript_engine_specific_sources(self) -> None:
        _, tokens, rank_of = toy_program()
        parents = recover_merge_parents(tokens, rank_of, base_token_count=3)
        sources = emit_regex_sources(
            tokens,
            parents,
            Compatibility.ECMASCRIPT,
            base_token_count=3,
        )
        self.assertIsInstance(sources, ECMAScriptRegexSources)
        validate_ecmascript_sources(sources, tokens, parents)
        self.assertEqual(len(sources.byte_rank_bits), 2)
        self.assertEqual(len(sources.merge_buckets), 3)
        self.assertLessEqual(sources.max_bucket_rules, 3)
        for source in sources.byte_rank_bits:
            self.assertNotIn("(?P<", source)
            self.assertNotIn("(?<", source)
        for source in sources.merge_buckets:
            self.assertNotIn("(?P<", source)
        self.assertTrue(any("(?<m" in source for source in sources.merge_buckets))

        encoded = encode_artifact(
            Encoding.R50K,
            Compatibility.ECMASCRIPT,
            sources,
            r"[a-z]+|.",
        )
        encoding, decoded, pretokenizer = decode_ecmascript_artifact(encoded)
        self.assertIs(encoding, Encoding.R50K)
        self.assertEqual(decoded, sources)
        self.assertEqual(pretokenizer, r"[a-z]+|.")

    def test_reserved_rank_is_skipped_by_both_emitters(self) -> None:
        tokens = (b"a", b"b", None, b"ab", b"aba")
        rank_of = {
            token: rank for rank, token in enumerate(tokens) if token is not None
        }
        parents = recover_merge_parents(tokens, rank_of, base_token_count=2)
        self.assertEqual(tuple(int(value) for value in parents[2]), (-1, -1))

        python_sources = emit_regex_sources(
            tokens,
            parents,
            Compatibility.PYTHON,
            base_token_count=2,
        )
        self.assertEqual(python_sources.reserved_ranks, (2,))
        program = RegexBPE(python_sources)
        match = program.fullmatch(b"aba")
        assert match is not None
        self.assertEqual(match.token_ids, [4])
        self.assertNotIn("m2", program.merge_pattern.groupindex)

        ecmascript_sources = emit_regex_sources(
            tokens,
            parents,
            Compatibility.ECMASCRIPT,
            base_token_count=2,
        )
        self.assertEqual(ecmascript_sources.reserved_ranks, (2,))
        validate_ecmascript_sources(ecmascript_sources, tokens, parents)


if __name__ == "__main__":
    unittest.main()
