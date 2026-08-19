import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isqrt
from typing import Any

from ..regex_ast import EMPTY, NEVER, RegexAST, render_regex
from ..tagged_fst import TaggedFST
from ._common import RANK_SEPARATOR, encode_rank

type ByteEscape = Callable[[int], str]


@dataclass(frozen=True, slots=True)
class RegexSources:
    byte_rank_bits: tuple[str, ...]
    merge_buckets: tuple[str, ...]
    merge_capture_ranks: tuple[tuple[int, ...], ...]
    merge_bucket_count: int
    token_count: int
    base_token_count: int
    rank_width: int
    reserved_ranks: tuple[int, ...] = ()

    @property
    def max_bucket_rules(self) -> int:
        return max(map(len, self.merge_capture_ranks), default=0)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _derive_merge_bucket_count(
    parent_pairs: Sequence[tuple[int, int]],
    token_count: int,
) -> int:
    """Balance dispatch width against the measured worst bucket width."""
    merge_rule_count = len(parent_pairs)
    if merge_rule_count == 0:
        return 1
    root = isqrt(merge_rule_count)
    lower_bound = root if root * root == merge_rule_count else root + 1
    candidate = max(2, lower_bound)
    while True:
        while not _is_prime(candidate):
            candidate += 1
        loads = [0] * candidate
        for left, right in parent_pairs:
            loads[merge_bucket_index(left, right, token_count, candidate)] += 1
        if max(loads) <= candidate:
            return candidate
        candidate += 1


def merge_bucket_index(
    left: int,
    right: int,
    token_count: int,
    bucket_count: int,
) -> int:
    """Map a complete parent pair to a bucket using exact integer arithmetic."""
    if not (0 <= left < token_count and 0 <= right < token_count):
        raise ValueError(f"parent ranks are outside the vocabulary: {(left, right)}")
    if bucket_count <= 0:
        raise ValueError("merge bucket count must be positive")
    return (left * token_count + right) % bucket_count


def _byte_escape(byte: int) -> str:
    return f"\\x{byte:02x}"


def _rank_stream_escape(byte: int) -> str:
    if byte == ord(",") or ord("0") <= byte <= ord("9"):
        return chr(byte)
    raise ValueError(f"unexpected rank-stream byte: {byte}")


def _emit_rank_bits(
    fst: TaggedFST,
    bit_count: int,
    *,
    escape: ByteEscape,
) -> tuple[str, ...]:
    def bit_output(bit: int) -> Callable[[int], RegexAST]:
        return lambda rank: EMPTY if rank & (1 << bit) else NEVER

    return tuple(
        render_regex(
            fst.to_regex(bit_output(bit)),
            escape_byte=escape,
        )
        for bit in range(bit_count)
    )


def _anonymous_capture(ranks: list[int], rank: int) -> str:
    ranks.append(rank)
    return "()"


def emit_sources(
    tokens: Sequence[bytes | None],
    parents: Any,
    *,
    base_token_count: int = 256,
) -> RegexSources:
    """Compile BPE ranks into ECMAScript membership and merge regexes."""
    token_count = len(tokens)
    if len(parents) != token_count:
        raise ValueError("tokens and parents must have the same length")
    if not 0 < base_token_count <= token_count:
        raise ValueError("invalid base token count")
    base_tokens = tokens[:base_token_count]
    if any(token is None or len(token) != 1 for token in base_tokens):
        raise ValueError("every base token must contain exactly one byte")
    if len(set(base_tokens)) != len(base_tokens):
        raise ValueError("base tokens must be unique")
    if token_count * token_count > (1 << 53) - 1:
        raise ValueError("vocabulary is too large for exact ECMAScript pair arithmetic")

    byte_fst = TaggedFST.from_pairs(
        (token, rank) for rank, token in enumerate(base_tokens) if token is not None
    )

    rank_width = max(1, len(str(token_count - 1)))
    merge_rules: list[tuple[int, int, int]] = []
    reserved_ranks: list[int] = []
    for child in range(base_token_count, token_count):
        if tokens[child] is None:
            reserved_ranks.append(child)
            if any(int(value) != -1 for value in parents[child]):
                raise ValueError(f"reserved rank {child} unexpectedly has parents")
            continue
        parent = parents[child]
        if len(parent) != 2:
            raise ValueError(f"rank {child} does not have two parents")
        left, right = (int(value) for value in parent)
        if not (0 <= left < child and 0 <= right < child):
            raise ValueError(f"rank {child} has invalid parents {(left, right)}")
        merge_rules.append((child, left, right))

    merge_bucket_count = _derive_merge_bucket_count(
        [(left, right) for _, left, right in merge_rules], token_count
    )
    merge_buckets: list[list[tuple[bytes, int]]] = [
        [] for _ in range(merge_bucket_count)
    ]
    for child, left, right in merge_rules:
        key = (
            encode_rank(left, rank_width)
            + RANK_SEPARATOR
            + encode_rank(right, rank_width)
        )
        bucket = merge_bucket_index(left, right, token_count, merge_bucket_count)
        merge_buckets[bucket].append((key, child))

    merge_capture_ranks: list[tuple[int, ...]] = []

    def render_bucket(pairs: list[tuple[bytes, int]]) -> str:
        ranks: list[int] = []
        source = render_regex(
            TaggedFST.from_pairs(pairs).to_regex(),
            escape_byte=_rank_stream_escape,
            emit_tag=lambda rank: _anonymous_capture(ranks, rank),
        )
        merge_capture_ranks.append(tuple(ranks))
        return source

    return RegexSources(
        byte_rank_bits=_emit_rank_bits(
            byte_fst,
            max(1, (base_token_count - 1).bit_length()),
            escape=_byte_escape,
        ),
        merge_buckets=tuple(render_bucket(pairs) for pairs in merge_buckets),
        merge_bucket_count=merge_bucket_count,
        token_count=token_count,
        base_token_count=base_token_count,
        rank_width=rank_width,
        reserved_ranks=tuple(reserved_ranks),
        merge_capture_ranks=tuple(merge_capture_ranks),
    )


def _compile_bit_patterns(sources: Sequence[str]) -> tuple[re.Pattern[bytes], ...]:
    patterns: list[re.Pattern[bytes]] = []
    for source in sources:
        try:
            pattern = re.compile(source.encode("ascii"))
        except UnicodeEncodeError as error:
            raise ValueError("ECMAScript byte patterns must be ASCII") from error
        if pattern.groups:
            raise ValueError("ECMAScript bit patterns must not contain captures")
        patterns.append(pattern)
    return tuple(patterns)


def _resolve_rank(patterns: Sequence[re.Pattern[bytes]], value: bytes) -> int:
    rank = 0
    for bit, pattern in enumerate(patterns):
        if pattern.fullmatch(value) is not None:
            rank |= 1 << bit
    return rank


def validate_sources(
    sources: RegexSources,
    tokens: Sequence[bytes | None],
    parents: Any,
) -> None:
    """Validate every embedded base rank and merge rule with stdlib ``re``."""
    byte_patterns = _compile_bit_patterns(sources.byte_rank_bits)
    if len(sources.merge_buckets) != sources.merge_bucket_count:
        raise ValueError("ECMAScript merge bucket count differs")
    if len(sources.merge_capture_ranks) != sources.merge_bucket_count:
        raise ValueError("ECMAScript merge capture table count differs")
    merge_patterns: list[re.Pattern[bytes]] = []
    capture_tables: list[tuple[int, ...]] = []
    for bucket, source in enumerate(sources.merge_buckets):
        pattern = re.compile(source.encode("ascii"))
        ranks = sources.merge_capture_ranks[bucket]
        if pattern.groupindex:
            raise ValueError("side-table ECMAScript bucket captures must be anonymous")
        if pattern.groups != len(ranks):
            raise ValueError("ECMAScript bucket capture table width differs")
        merge_patterns.append(pattern)
        capture_tables.append(ranks)

    expected_merge_ranks = set(range(sources.base_token_count, sources.token_count))
    expected_merge_ranks.difference_update(sources.reserved_ranks)
    actual_merge_ranks = {
        rank for capture_table in capture_tables for rank in capture_table
    }
    if actual_merge_ranks != expected_merge_ranks or sum(
        map(len, capture_tables)
    ) != len(expected_merge_ranks):
        raise ValueError("ECMAScript merge capture ranks differ")
    for expected_rank, token in enumerate(tokens[: sources.base_token_count]):
        if token is None:
            raise ValueError(f"base rank {expected_rank} is reserved")
        actual_rank = _resolve_rank(byte_patterns, token)
        if actual_rank != expected_rank:
            raise ValueError(
                f"ECMAScript byte rank differs: {actual_rank} != {expected_rank}"
            )
    reserved = set(sources.reserved_ranks)
    for expected_rank in range(sources.base_token_count, sources.token_count):
        if expected_rank in reserved:
            continue
        left, right = (int(value) for value in parents[expected_rank])
        bucket = merge_bucket_index(
            left,
            right,
            sources.token_count,
            sources.merge_bucket_count,
        )
        pattern = merge_patterns[bucket]
        pair = (
            encode_rank(left, sources.rank_width)
            + RANK_SEPARATOR
            + encode_rank(right, sources.rank_width)
        )
        match = pattern.fullmatch(pair)
        if match is None or match.lastindex is None:
            raise ValueError(
                f"ECMAScript merge rule is missing for rank {expected_rank}"
            )
        actual_rank = capture_tables[bucket][match.lastindex - 1]
        if actual_rank != expected_rank:
            raise ValueError(
                f"ECMAScript merge rank differs: {actual_rank} != {expected_rank}"
            )
