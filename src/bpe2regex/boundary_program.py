"""Runtime for a canonical-boundary regex plus a token-rank lookup regex."""

import re
from collections.abc import Mapping
from typing import Any, Self

from .match import TokenMatch
from .reir.automata.canonical_boundary import CanonicalBoundaryRegexSource


class BoundaryRegexBPE:
    """Emit canonical BPE tokens using only repeated regex full matches."""

    def __init__(
        self,
        source: CanonicalBoundaryRegexSource,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.metadata = dict(metadata or {})
        try:
            boundary = source.boundary_pattern.encode("ascii")
            token_lookup = source.token_to_rank.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(
                "canonical regex source must contain ASCII syntax"
            ) from error
        self.boundary_pattern = re.compile(boundary)
        self.token_pattern = re.compile(token_lookup)
        if self.boundary_pattern.groupindex or self.token_pattern.groupindex:
            raise ValueError("canonical regex captures must be anonymous")
        if self.boundary_pattern.groups != source.boundary_capture_count:
            raise ValueError("boundary capture count differs from its pattern")
        if self.token_pattern.groups != len(source.token_capture_ranks):
            raise ValueError("token capture table width differs from its pattern")
        if len(set(source.token_capture_ranks)) != len(source.token_capture_ranks):
            raise ValueError("token capture table contains duplicate ranks")
        self._token_ranks = (-1, *source.token_capture_ranks)
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
            match = self.boundary_pattern.fullmatch(source, position)
            if match is None:
                return None
            participating = tuple(
                group
                for group in range(1, self.boundary_pattern.groups + 1)
                if match.start(group) >= 0
            )
            if len(participating) != 1:
                raise ValueError(
                    "canonical boundary regex must select exactly one capture"
                )
            boundary = match.start(participating[0])
            if boundary <= position:
                raise ValueError("canonical regex did not advance its token boundary")
            rank_match = self.token_pattern.fullmatch(source, position, boundary)
            if rank_match is None or rank_match.lastindex is None:
                raise ValueError("canonical token boundary has no rank lookup")
            rank = self._token_ranks[rank_match.lastindex]
            if rank < 0:
                raise ValueError("token lookup selected an unknown rank capture")
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


__all__ = ["BoundaryRegexBPE"]
