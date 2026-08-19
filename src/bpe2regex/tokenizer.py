from typing import Protocol, Self

from .encoding import Encoding
from .match import TokenMatch
from .pretokenize import PreTokenizer


class BytePattern(Protocol):
    def fullmatch(self, piece: bytes | bytearray | memoryview) -> TokenMatch | None: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...


def repair_surrogates(text: str) -> str:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return text


class Tokenizer[EncodingT: Encoding]:
    """Ordinary-text tokenizer parameterized by its BPE encoding variant."""

    def __init__(
        self,
        pattern: BytePattern,
        pretokenizer: PreTokenizer[EncodingT],
    ) -> None:
        self.encoding = pretokenizer.encoding
        self.pattern = pattern
        self.pretokenizer = pretokenizer

    def encode_ordinary(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode_ordinary() requires str")
        text = repair_surrogates(text)
        token_ids: list[int] = []
        for pretoken in self.pretokenizer.finditer(text):
            match = self.pattern.fullmatch(pretoken.text.encode("utf-8"))
            if match is None:
                raise RuntimeError(
                    f"{self.encoding.value} byte matcher rejected a pre-tokenizer piece"
                )
            token_ids.extend(match.token_ids)
        return token_ids

    def close(self) -> None:
        self.pattern.close()

    def __enter__(self) -> Self:
        self.pattern.__enter__()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
