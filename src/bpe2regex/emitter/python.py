from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..rank_codec import RANK_ALPHABET, encode_rank_pair, rank_code_width
from ..regex_ast import render_regex
from ..tagged_fst import TaggedFST


@dataclass(frozen=True, slots=True)
class RegexSources:
    byte_to_rank: str
    byte_capture_ranks: tuple[int, ...]
    merge_pair: str
    merge_capture_ranks: tuple[int, ...]
    token_count: int
    base_token_count: int
    rank_width: int
    reserved_ranks: tuple[int, ...] = ()


def _byte_escape(byte: int) -> str:
    return f"\\x{byte:02x}"


def _rank_stream_escape(byte: int) -> str:
    if byte in RANK_ALPHABET:
        return chr(byte)
    raise ValueError(f"unexpected rank-stream byte: {byte}")


def _anonymous_capture(ranks: list[int], rank: int) -> str:
    ranks.append(rank)
    return "()"


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

    rank_width = rank_code_width(token_count)
    byte_fst = TaggedFST.from_pairs(
        # The validation above proves that every base token is bytes.
        (token, rank)
        for rank, token in enumerate(base_tokens)
        if token is not None
    )
    byte_capture_ranks: list[int] = []
    byte_to_rank = render_regex(
        byte_fst.to_regex(),
        escape_byte=_byte_escape,
        emit_tag=lambda rank: _anonymous_capture(byte_capture_ranks, rank),
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
        key = encode_rank_pair(left, right, rank_width)
        merge_pairs.append((key, child))

    merge_fst = TaggedFST.from_pairs(merge_pairs)
    merge_capture_ranks: list[int] = []
    merge_pair = render_regex(
        merge_fst.to_regex(),
        escape_byte=_rank_stream_escape,
        emit_tag=lambda rank: _anonymous_capture(merge_capture_ranks, rank),
    )
    return RegexSources(
        byte_to_rank=byte_to_rank,
        merge_pair=merge_pair,
        token_count=token_count,
        base_token_count=base_token_count,
        rank_width=rank_width,
        reserved_ranks=tuple(reserved_ranks),
        byte_capture_ranks=tuple(byte_capture_ranks),
        merge_capture_ranks=tuple(merge_capture_ranks),
    )
