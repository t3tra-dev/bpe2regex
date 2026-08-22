import re
from collections.abc import Mapping
from typing import Any, Self

from .match import TokenMatch
from .reir.automata import CanonicalTokenRegexSource


class CanonicalRegexBPE:
    """Tokenizer whose control flow is entirely encoded in one regex."""

    def __init__(
        self,
        source: CanonicalTokenRegexSource,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.metadata = dict(metadata or {})
        try:
            pattern_source = source.pattern.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(
                "canonical regex source must contain ASCII syntax"
            ) from error
        self.pattern = re.compile(pattern_source)
        if self.pattern.groupindex:
            raise ValueError("canonical regex captures must be anonymous")
        if self.pattern.groups != len(source.capture_ranks):
            raise ValueError("canonical capture table width differs from its pattern")
        self._capture_ranks = (-1, *source.capture_ranks)
        self._closed = False

    def fullmatch(self, piece: bytes | bytearray | memoryview) -> TokenMatch | None:
        if self._closed:
            raise RuntimeError("regex program is closed")
        if not isinstance(piece, (bytes, bytearray, memoryview)):
            raise TypeError("fullmatch() requires a bytes-like object")
        source = bytes(piece)
        position = 0
        token_ids: list[int] = []
        spans: list[tuple[int, int]] = []
        while position < len(source):
            match = self.pattern.fullmatch(source, position)
            if match is None or match.lastindex is None:
                return None
            boundary = match.start(match.lastindex)
            if boundary <= position:
                raise ValueError("canonical regex did not advance its token boundary")
            rank = self._capture_ranks[match.lastindex]
            if rank < 0:
                raise ValueError("canonical regex selected an unknown rank capture")
            token_ids.append(rank)
            spans.append((position, boundary))
            position = boundary
        return TokenMatch(
            source,
            token_ids=token_ids,
            _token_spans=spans,
            path_count=1,
        )

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("regex program is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["CanonicalRegexBPE"]
