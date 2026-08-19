import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ._common import RANK_SEPARATOR, encode_rank

BASE_GROUP_PREFIX = "b"
MERGE_GROUP_PREFIX = "m"


@dataclass(slots=True)
class _TrieNode:
    children: dict[int, _TrieNode] = field(default_factory=dict)
    terminal_rank: int | None = None


@dataclass(frozen=True, slots=True)
class RegexSources:
    byte_to_rank: str
    merge_pair: str
    token_count: int
    base_token_count: int
    rank_width: int
    reserved_ranks: tuple[int, ...] = ()


def _insert_trie(root: _TrieNode, key: bytes, rank: int) -> None:
    node = root
    for byte in key:
        node = node.children.setdefault(byte, _TrieNode())
    if node.terminal_rank is not None:
        raise ValueError(
            f"duplicate merge parent pair for ranks {node.terminal_rank}, {rank}"
        )
    node.terminal_rank = rank


def _emit_trie(node: _TrieNode, *, group_prefix: str) -> bytes:
    alternatives: list[bytes] = []
    if node.terminal_rank is not None:
        alternatives.append(f"(?P<{group_prefix}{node.terminal_rank}>)".encode("ascii"))
    for byte, child in sorted(node.children.items()):
        alternatives.append(
            re.escape(bytes((byte,))) + _emit_trie(child, group_prefix=group_prefix)
        )
    if not alternatives:
        raise ValueError("cannot emit an empty trie node")
    if len(alternatives) == 1:
        return alternatives[0]
    return b"(?:" + b"|".join(alternatives) + b")"


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
    byte_alternatives = [
        # The validation above proves that every base token is bytes.
        f"(?P<{BASE_GROUP_PREFIX}{rank}>\\x{token[0]:02x})"
        for rank, token in enumerate(base_tokens)
        if token is not None
    ]
    byte_to_rank = "(?:" + "|".join(byte_alternatives) + ")"

    root = _TrieNode()
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
        _insert_trie(root, key, child)

    merge_pair = _emit_trie(root, group_prefix=MERGE_GROUP_PREFIX).decode("ascii")
    return RegexSources(
        byte_to_rank=byte_to_rank,
        merge_pair=merge_pair,
        token_count=token_count,
        base_token_count=base_token_count,
        rank_width=rank_width,
        reserved_ranks=tuple(reserved_ranks),
    )
