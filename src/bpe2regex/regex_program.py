import heapq
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from .binary import PYTHON_ARTIFACT_FILENAME, decode_python_artifact
from .emitter._common import RANK_SEPARATOR, encode_rank
from .emitter.python import RegexSources as PythonRegexSources
from .encoding import Encoding
from .match import TokenMatch

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


def _group_rank_table(
    pattern: re.Pattern[bytes],
    *,
    expected_ranks: Iterable[int],
    capture_ranks: tuple[int, ...],
) -> tuple[int, ...]:
    expected = set(expected_ranks)
    if pattern.groupindex:
        raise ValueError("side-table regex captures must be anonymous")
    if pattern.groups != len(capture_ranks):
        raise ValueError("regex capture table width differs from its pattern")
    if len(set(capture_ranks)) != len(capture_ranks):
        raise ValueError("regex capture table contains duplicate ranks")
    if set(capture_ranks) != expected:
        missing = sorted(expected - set(capture_ranks))[:5]
        extra = sorted(set(capture_ranks) - expected)[:5]
        raise ValueError(
            f"rank capture table differs; missing={missing}, extra={extra}"
        )
    return (-1, *capture_ranks)


class RegexBPE:
    """Exact ranked-BPE rewrite program."""

    def __init__(
        self,
        sources: PythonRegexSources,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.sources = sources
        self.metadata = dict(metadata or {})
        try:
            byte_source = sources.byte_to_rank.encode("ascii")
            merge_source = sources.merge_pair.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(
                "byte regex sources must contain ASCII syntax only"
            ) from error

        self.byte_pattern = re.compile(byte_source)
        self.merge_pattern = re.compile(merge_source)
        self._base_ranks = _group_rank_table(
            self.byte_pattern,
            expected_ranks=range(sources.base_token_count),
            capture_ranks=sources.byte_capture_ranks,
        )
        self._merge_ranks = _group_rank_table(
            self.merge_pattern,
            expected_ranks=(
                rank
                for rank in range(sources.base_token_count, sources.token_count)
                if rank not in sources.reserved_ranks
            ),
            capture_ranks=sources.merge_capture_ranks,
        )
        if tuple(sorted(set(sources.reserved_ranks))) != sources.reserved_ranks or any(
            rank < sources.base_token_count or rank >= sources.token_count
            for rank in sources.reserved_ranks
        ):
            raise ValueError("reserved ranks are not canonical for the regex program")
        if sources.rank_width != max(1, len(str(sources.token_count - 1))):
            raise ValueError("rank width is not canonical for the token count")
        if sources.base_token_count == 256:
            observed_ranks: set[int] = set()
            for value in range(256):
                match = self.byte_pattern.fullmatch(bytes((value,)))
                if match is None or match.lastindex is None:
                    raise ValueError(f"byte pattern does not cover byte {value}")
                observed_ranks.add(self._base_ranks[match.lastindex])
            if observed_ranks != set(range(256)):
                raise ValueError(
                    "byte pattern does not map bytes bijectively to base ranks"
                )
        self._merge_rank_cache: dict[tuple[int, int], int] = {}
        self._closed = False

    def _initial_ranks(self, source: bytes) -> list[int] | None:
        ranks: list[int] = []
        position = 0
        for match in self.byte_pattern.finditer(source):
            if match.start() != position or match.end() != position + 1:
                return None
            if match.lastindex is None:
                raise ValueError("byte pattern matched without a rank capture")
            rank = self._base_ranks[match.lastindex]
            if rank < 0:
                raise ValueError("byte pattern selected an unknown rank capture")
            ranks.append(rank)
            position += 1
        return ranks if position == len(source) else None

    def _merge_rank(self, left: int, right: int) -> int | None:
        key = (left, right)
        cached = self._merge_rank_cache.get(key)
        if cached is not None:
            return cached
        encoded = (
            encode_rank(left, self.sources.rank_width)
            + RANK_SEPARATOR
            + encode_rank(right, self.sources.rank_width)
        )
        match = self.merge_pattern.fullmatch(encoded)
        if match is None:
            return None
        if match.lastindex is None:
            raise ValueError("merge pattern matched without a rank capture")
        rank = self._merge_ranks[match.lastindex]
        if rank < 0:
            raise ValueError("merge pattern selected an unknown rank capture")
        self._merge_rank_cache[key] = rank
        return rank

    def fullmatch(self, piece: bytes | bytearray | memoryview) -> TokenMatch | None:
        if self._closed:
            raise RuntimeError("regex program is closed")
        if not isinstance(piece, (bytes, bytearray, memoryview)):
            raise TypeError("fullmatch() requires a bytes-like object")
        source = bytes(piece)
        token_ids = self._initial_ranks(source)
        if token_ids is None:
            return None
        if not token_ids:
            return TokenMatch(source, token_ids=[], _token_spans=[], path_count=1)

        token_count = len(token_ids)
        starts = list(range(token_count))
        ends = list(range(1, token_count + 1))
        previous = [-1, *range(token_count - 1)]
        following = [*range(1, token_count), -1]
        versions = [0] * token_count
        candidates: list[tuple[int, int, int, int, int]] = []

        def push_candidate(left_index: int) -> None:
            right_index = following[left_index]
            if right_index < 0:
                return
            rank = self._merge_rank(token_ids[left_index], token_ids[right_index])
            if rank is not None:
                heapq.heappush(
                    candidates,
                    (
                        rank,
                        left_index,
                        versions[left_index],
                        right_index,
                        versions[right_index],
                    ),
                )

        for left_index in range(token_count - 1):
            push_candidate(left_index)

        while candidates:
            rank, left_index, left_version, right_index, right_version = heapq.heappop(
                candidates
            )
            if (
                versions[left_index] != left_version
                or versions[right_index] != right_version
                or following[left_index] != right_index
                or previous[right_index] != left_index
            ):
                continue

            token_ids[left_index] = rank
            ends[left_index] = ends[right_index]
            next_index = following[right_index]
            following[left_index] = next_index
            if next_index >= 0:
                previous[next_index] = left_index
            following[right_index] = -2
            previous[right_index] = -2
            versions[left_index] += 1
            versions[right_index] += 1

            previous_index = previous[left_index]
            if previous_index >= 0:
                push_candidate(previous_index)
            push_candidate(left_index)

        final_ids: list[int] = []
        final_spans: list[tuple[int, int]] = []
        index = 0
        while index >= 0:
            final_ids.append(token_ids[index])
            final_spans.append((starts[index], ends[index]))
            index = following[index]
        return TokenMatch(
            source,
            token_ids=final_ids,
            _token_spans=final_spans,
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


def load_byte_pattern(
    encoding: Encoding,
    artifact: str | Path,
) -> RegexBPE:
    path = Path(artifact)
    if path.is_dir():
        path /= PYTHON_ARTIFACT_FILENAME
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read regex program artifact: {path}") from error
    artifact_encoding, sources, _ = decode_python_artifact(content)
    if artifact_encoding is not encoding:
        raise ValueError(
            f"artifact encoding {artifact_encoding!r} does not match {encoding!r}"
        )
    _validate_sources(encoding, sources)
    return RegexBPE(sources, metadata={"encoding": artifact_encoding.value})


def load_tokenizer[EncodingT: Encoding](
    encoding: EncodingT,
    artifact: str | Path,
) -> Tokenizer[EncodingT]:
    from .pretokenize import PreTokenizer
    from .tokenizer import Tokenizer

    path = Path(artifact)
    if path.is_dir():
        path /= PYTHON_ARTIFACT_FILENAME
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read regex program artifact: {path}") from error
    artifact_encoding, sources, pretokenizer_source = decode_python_artifact(content)
    if artifact_encoding is not encoding:
        raise ValueError(
            f"artifact encoding {artifact_encoding!r} does not match {encoding!r}"
        )
    _validate_sources(encoding, sources)
    pattern = RegexBPE(
        sources,
        metadata={"encoding": artifact_encoding.value},
    )
    try:
        pretokenizer = PreTokenizer.from_source(
            encoding,
            pretokenizer_source,
            metadata={"source": "embedded regex program"},
        )
        return Tokenizer(pattern, pretokenizer)
    except BaseException:
        pattern.close()
        raise


def _validate_sources(encoding: Encoding, sources: PythonRegexSources) -> None:
    expected = (
        encoding.token_count,
        encoding.base_token_count,
        encoding.rank_width,
        encoding.reserved_ranks,
    )
    actual = (
        sources.token_count,
        sources.base_token_count,
        sources.rank_width,
        sources.reserved_ranks,
    )
    if actual != expected:
        raise ValueError(f"unexpected {encoding.value} regex dimensions: {actual!r}")
