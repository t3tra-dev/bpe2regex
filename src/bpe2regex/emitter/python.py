from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..regex_ast import render_regex
from ..tagged_fst import TaggedFST
from ._common import RANK_SEPARATOR, encode_rank

BASE_GROUP_PREFIX = "b"
MERGE_GROUP_PREFIX = "m"


@dataclass(frozen=True, slots=True)
class RegexSources:
    byte_to_rank: str
    merge_pair: str
    token_count: int
    base_token_count: int
    rank_width: int
    reserved_ranks: tuple[int, ...] = ()


def _byte_escape(byte: int) -> str:
    return f"\\x{byte:02x}"


def _rank_stream_escape(byte: int) -> str:
    if byte == ord(",") or ord("0") <= byte <= ord("9"):
        return chr(byte)
    raise ValueError(f"unexpected rank-stream byte: {byte}")


def _capture(group_prefix: str, rank: int) -> str:
    return f"(?P<{group_prefix}{rank}>)"


def emit_sources(
    tokens: Sequence[bytes | None],
    parents: Any,
    *,
    base_token_count: int = 256,
) -> RegexSources:
    """Compile ranked BPE data into Python stdlib ``re`` sources."""
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

    rank_width = max(1, len(str(token_count - 1)))
    byte_fst = TaggedFST.from_pairs(
        # The validation above proves that every base token is bytes.
        (token, rank)
        for rank, token in enumerate(base_tokens)
        if token is not None
    )
    byte_to_rank = render_regex(
        byte_fst.to_regex(),
        escape_byte=_byte_escape,
        emit_tag=lambda rank: _capture(BASE_GROUP_PREFIX, rank),
    )

    merge_pairs: list[tuple[bytes, int]] = []
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
        key = (
            encode_rank(left, rank_width)
            + RANK_SEPARATOR
            + encode_rank(right, rank_width)
        )
        merge_pairs.append((key, child))

    merge_fst = TaggedFST.from_pairs(merge_pairs)
    merge_pair = render_regex(
        merge_fst.to_regex(),
        escape_byte=_rank_stream_escape,
        emit_tag=lambda rank: _capture(MERGE_GROUP_PREFIX, rank),
    )
    return RegexSources(
        byte_to_rank=byte_to_rank,
        merge_pair=merge_pair,
        token_count=token_count,
        base_token_count=base_token_count,
        rank_width=rank_width,
        reserved_ranks=tuple(reserved_ranks),
    )
