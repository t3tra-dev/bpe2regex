import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Self, get_args

from bpe2regex import (
    CL100K,
    O200K,
    P50K,
    R50K,
    BuildResult,
    Encoding,
    PreTokenizer,
    Tokenizer,
    TokenMatch,
    build_regex_artifact,
)

if TYPE_CHECKING:
    _built: BuildResult[R50K] = build_regex_artifact(Encoding.R50K)


class _BytePattern:
    def fullmatch(
        self,
        piece: bytes | bytearray | memoryview,
    ) -> TokenMatch | None:
        source = bytes(piece)
        return TokenMatch(
            source,
            token_ids=list(source),
            _token_spans=[(index, index + 1) for index in range(len(source))],
            path_count=1,
        )

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self


class EncodingAbstractionTests(unittest.TestCase):
    def test_r50k_literal_and_generic_runtime_types(self) -> None:
        self.assertEqual(get_args(R50K.__value__), (Encoding.R50K,))

        pretokenizer: PreTokenizer[R50K] = PreTokenizer.from_source(
            Encoding.R50K,
            r"[\s\S]",
        )
        tokenizer: Tokenizer[R50K] = Tokenizer(_BytePattern(), pretokenizer)
        self.assertIs(tokenizer.encoding, Encoding.R50K)
        self.assertEqual(tokenizer.encode_ordinary("ab"), [97, 98])

        result: BuildResult[R50K] = BuildResult(
            encoding=Encoding.R50K,
            directory=Path("artifact"),
            metadata={},
        )
        self.assertIs(result.encoding, Encoding.R50K)

    def test_o200k_literal_type(self) -> None:
        self.assertEqual(get_args(O200K.__value__), (Encoding.O200K,))

        result: BuildResult[O200K] = BuildResult(
            encoding=Encoding.O200K,
            directory=Path("artifact"),
            metadata={},
        )
        self.assertIs(result.encoding, Encoding.O200K)

    def test_p50k_literal_and_reserved_rank(self) -> None:
        self.assertEqual(get_args(P50K.__value__), (Encoding.P50K,))
        self.assertEqual(Encoding.P50K.token_count, 50_281)
        self.assertEqual(Encoding.P50K.mergeable_token_count, 50_280)
        self.assertEqual(Encoding.P50K.reserved_ranks, (50_256,))

        result: BuildResult[P50K] = BuildResult(
            encoding=Encoding.P50K,
            directory=Path("artifact"),
            metadata={},
        )
        self.assertIs(result.encoding, Encoding.P50K)

    def test_cl100k_literal_type(self) -> None:
        self.assertEqual(get_args(CL100K.__value__), (Encoding.CL100K,))

        result: BuildResult[CL100K] = BuildResult(
            encoding=Encoding.CL100K,
            directory=Path("artifact"),
            metadata={},
        )
        self.assertIs(result.encoding, Encoding.CL100K)


if __name__ == "__main__":
    unittest.main()
