import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isqrt
from typing import Any

from ...rank_codec import RANK_ALPHABET, encode_rank_pair, rank_code_width
from ...tagged_fst import TaggedFST
from ..ops import EPSILON, NEVER, Op
from ..source import render_regex
from ..tagged_source import render_tagged_regex

type ByteEscape = Callable[[int], str]


@dataclass(frozen=True, slots=True)
class RegexSources:
    byte_rank_bits: tuple[str, ...]
    merge_prefixes: tuple[str, ...]
    merge_patterns: tuple[str, ...]
    merge_capture_ranks: tuple[tuple[int, ...], ...]
    token_count: int
    base_token_count: int
    rank_width: int
    reserved_ranks: tuple[int, ...] = ()

    @property
    def max_pattern_rules(self) -> int:
        return max(map(len, self.merge_capture_ranks), default=0)


@dataclass(frozen=True, slots=True)
class _FrontierFragment:
    prefix: bytes
    state: int
    rule_count: int


def _uint_width(value: int) -> int:
    width = 1
    while value >= 0x80:
        value >>= 7
        width += 1
    return width


def _ceil_sqrt(value: int) -> int:
    root = isqrt(value)
    return root if root * root == value else root + 1


def _split_merge_frontier(
    fst: TaggedFST,
    *,
    key_width: int,
    max_rules: int,
) -> tuple[_FrontierFragment, ...]:
    """Choose a minimum-local-cost, prefix-free trie frontier with a rule cap."""
    if key_width <= 0:
        raise ValueError("merge trie key width must be positive")
    if max_rules <= 0:
        raise ValueError("merge frontier rule bound must be positive")

    state_count = len(fst.states)
    prefixes: list[bytes | None] = [None] * state_count
    prefixes[fst.start] = b""
    pending = [fst.start]
    while pending:
        state_index = pending.pop()
        prefix = prefixes[state_index]
        assert prefix is not None
        state = fst.states[state_index]
        if state.output is not None and len(prefix) != key_width:
            raise ValueError("merge trie has a terminal outside the fixed key width")
        if len(prefix) >= key_width and state.transitions:
            raise ValueError("merge trie has transitions beyond the fixed key width")
        for symbol, target in state.transitions:
            if prefixes[target] is not None:
                raise ValueError("merge frontier requires a trie without shared states")
            prefixes[target] = prefix + bytes((symbol,))
            pending.append(target)
    if any(prefix is None for prefix in prefixes):
        raise ValueError("merge trie contains an unreachable state")

    output_counts: list[int | None] = [None] * state_count
    source_lengths: list[int | None] = [None] * state_count

    def measure(state_index: int) -> tuple[int, int]:
        known_count = output_counts[state_index]
        known_length = source_lengths[state_index]
        if known_count is not None and known_length is not None:
            return known_count, known_length
        state = fst.states[state_index]
        output_count = int(state.output is not None)
        branch_lengths = [2] if state.output is not None else []
        for _, target in state.transitions:
            child_count, child_length = measure(target)
            output_count += child_count
            branch_lengths.append(1 + child_length)
        if not branch_lengths:
            source_length = 4  # ``(?!)``
        elif len(branch_lengths) == 1:
            source_length = branch_lengths[0]
        else:
            source_length = sum(branch_lengths) + len(branch_lengths) + 3
        output_counts[state_index] = output_count
        source_lengths[state_index] = source_length
        return output_count, source_length

    measure(fst.start)
    plan_costs = [0] * state_count
    plan_counts = [0] * state_count
    keep_state = [False] * state_count

    def plan(state_index: int) -> tuple[int, int]:
        state = fst.states[state_index]
        prefix = prefixes[state_index]
        output_count = output_counts[state_index]
        source_length = source_lengths[state_index]
        assert prefix is not None
        assert output_count is not None
        assert source_length is not None
        if output_count == 0:
            return 0, 0

        split_cost = 0
        split_count = 0
        split_allowed = state.output is None and bool(state.transitions)
        if split_allowed:
            for _, target in state.transitions:
                child_cost, child_count = plan(target)
                split_cost += child_cost
                split_count += child_count

        keep_allowed = output_count <= max_rules
        keep_cost = (
            _uint_width(len(prefix))
            + len(prefix)
            + _uint_width(source_length)
            + source_length
            + _uint_width(output_count)
        )
        if keep_allowed and (
            not split_allowed or (keep_cost, 1, 0) <= (split_cost, split_count, 1)
        ):
            plan_costs[state_index] = keep_cost
            plan_counts[state_index] = 1
            keep_state[state_index] = True
        elif split_allowed:
            plan_costs[state_index] = split_cost
            plan_counts[state_index] = split_count
        else:
            raise ValueError("merge trie cannot satisfy the frontier rule bound")
        return plan_costs[state_index], plan_counts[state_index]

    plan(fst.start)
    fragments: list[_FrontierFragment] = []

    def collect(state_index: int) -> None:
        output_count = output_counts[state_index]
        assert output_count is not None
        if output_count == 0:
            return
        if keep_state[state_index]:
            prefix = prefixes[state_index]
            assert prefix is not None
            fragments.append(_FrontierFragment(prefix, state_index, output_count))
            return
        for _, target in fst.states[state_index].transitions:
            collect(target)

    collect(fst.start)
    return tuple(fragments)


def _byte_escape(byte: int) -> str:
    return f"\\x{byte:02x}"


def _rank_stream_escape(byte: int) -> str:
    if byte in RANK_ALPHABET:
        return chr(byte)
    raise ValueError(f"unexpected rank-stream byte: {byte}")


def _emit_rank_bits(
    fst: TaggedFST,
    bit_count: int,
    *,
    escape: ByteEscape,
) -> tuple[str, ...]:
    def bit_output(bit: int) -> Callable[[int], Op]:
        return lambda rank: EPSILON if rank & (1 << bit) else NEVER

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
    byte_fst = TaggedFST.from_pairs(
        (token, rank) for rank, token in enumerate(base_tokens) if token is not None
    )

    rank_width = rank_code_width(token_count)
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

    merge_pairs = [
        (encode_rank_pair(left, right, rank_width), child)
        for child, left, right in merge_rules
    ]
    merge_fst = TaggedFST.from_pairs(merge_pairs)
    merge_rule_count = len(merge_pairs)
    frontier = (
        _split_merge_frontier(
            merge_fst,
            key_width=rank_width * 2,
            max_rules=_ceil_sqrt(merge_rule_count),
        )
        if merge_rule_count
        else ()
    )
    merge_capture_ranks: list[tuple[int, ...]] = []

    def render_fragment(fragment: _FrontierFragment) -> str:
        ranks: list[int] = []
        source = render_tagged_regex(
            merge_fst.to_regex(start=fragment.state),
            escape_byte=_rank_stream_escape,
            emit_tag=lambda rank: _anonymous_capture(ranks, rank),
        )
        if len(ranks) != fragment.rule_count:
            raise ValueError("merge frontier output count changed while rendering")
        merge_capture_ranks.append(tuple(ranks))
        return source

    return RegexSources(
        byte_rank_bits=_emit_rank_bits(
            byte_fst,
            max(1, (base_token_count - 1).bit_length()),
            escape=_byte_escape,
        ),
        merge_prefixes=tuple(fragment.prefix.decode("ascii") for fragment in frontier),
        merge_patterns=tuple(render_fragment(fragment) for fragment in frontier),
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
    frontier_count = len(sources.merge_prefixes)
    if len(sources.merge_patterns) != frontier_count:
        raise ValueError("ECMAScript merge frontier pattern count differs")
    if len(sources.merge_capture_ranks) != frontier_count:
        raise ValueError("ECMAScript merge capture table count differs")
    encoded_prefixes: list[bytes] = []
    for prefix in sources.merge_prefixes:
        try:
            encoded = prefix.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("ECMAScript merge prefixes must be ASCII") from error
        if len(encoded) > sources.rank_width * 2 or any(
            byte not in RANK_ALPHABET for byte in encoded
        ):
            raise ValueError("ECMAScript merge prefix is not canonical base62")
        encoded_prefixes.append(encoded)
    sorted_prefixes = sorted(encoded_prefixes)
    if len(set(encoded_prefixes)) != frontier_count or any(
        right.startswith(left) for left, right in pairwise(sorted_prefixes)
    ):
        raise ValueError("ECMAScript merge frontier is not prefix-free")
    prefix_to_pattern = {prefix: index for index, prefix in enumerate(encoded_prefixes)}
    merge_patterns: list[re.Pattern[bytes]] = []
    capture_tables: list[tuple[int, ...]] = []
    for pattern_index, source in enumerate(sources.merge_patterns):
        pattern = re.compile(source.encode("ascii"))
        ranks = sources.merge_capture_ranks[pattern_index]
        if pattern.groupindex:
            raise ValueError("side-table ECMAScript captures must be anonymous")
        if pattern.groups != len(ranks):
            raise ValueError("ECMAScript merge capture table width differs")
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
        pair = encode_rank_pair(left, right, sources.rank_width)
        dispatched = next(
            (
                (prefix_to_pattern[pair[:length]], length)
                for length in range(len(pair) + 1)
                if pair[:length] in prefix_to_pattern
            ),
            None,
        )
        if dispatched is None:
            raise ValueError(
                f"ECMAScript merge frontier is missing rank {expected_rank}"
            )
        pattern_index, prefix_length = dispatched
        match = merge_patterns[pattern_index].fullmatch(pair[prefix_length:])
        if match is None or match.lastindex is None:
            raise ValueError(
                f"ECMAScript merge rule is missing for rank {expected_rank}"
            )
        actual_rank = capture_tables[pattern_index][match.lastindex - 1]
        if actual_rank != expected_rank:
            raise ValueError(
                f"ECMAScript merge rank differs: {actual_rank} != {expected_rank}"
            )
