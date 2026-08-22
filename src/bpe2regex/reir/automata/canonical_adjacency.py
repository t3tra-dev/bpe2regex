"""Persistent 1-local adjacency representation for canonical BPE tokens."""

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .ir import DFA, Transition
from .labels import SymbolSet


@dataclass(frozen=True, slots=True)
class CanonicalAdjacencyMetrics:
    applied_merges: int
    state_count: int
    active_token_count: int
    state_parent_links: int
    token_parent_links: int
    exclusion_count: int
    dense_cell_count: int
    persistent_record_count: int
    elapsed_seconds: float


@dataclass(slots=True)
class CanonicalAdjacencyIR:
    """A persistent-clone encoding of the universal canonical-token DFA.

    BPE's universal canonical-token automaton remains 1-local: each token has
    one global target state.  A merge clones one token column and one state
    row, then records at most two denied cells.  Birth order determines whether
    a cell observes the row clone or the token clone first.
    """

    alphabet_size: int
    base_token_count: int
    state_parents: tuple[int | None, ...]
    state_births: tuple[int, ...]
    state_exclusions: tuple[frozenset[int], ...]
    token_parents: tuple[int | None, ...]
    token_births: tuple[int, ...]
    token_targets: tuple[int | None, ...]
    applied_merges: int
    _allowed_cache: dict[tuple[int, int], bool] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not 0 < self.base_token_count <= self.alphabet_size:
            raise ValueError("invalid persistent adjacency base token count")
        if not self.state_parents:
            raise ValueError("persistent adjacency requires a base state")
        if not (
            len(self.state_parents)
            == len(self.state_births)
            == len(self.state_exclusions)
        ):
            raise ValueError("persistent adjacency state columns differ in length")
        if not (
            len(self.token_parents)
            == len(self.token_births)
            == len(self.token_targets)
            == self.alphabet_size
        ):
            raise ValueError("persistent adjacency token columns differ in length")
        if self.state_parents[0] is not None or self.state_births[0] != 0:
            raise ValueError("persistent adjacency base state is not canonical")
        for state in range(1, self.state_count):
            parent = self.state_parents[state]
            if parent is None or not 0 <= parent < state:
                raise ValueError(f"state {state} has an invalid clone parent")
            if self.state_births[state] <= self.state_births[parent]:
                raise ValueError(f"state {state} does not postdate its parent")
        for token in range(self.alphabet_size):
            birth = self.token_births[token]
            target = self.token_targets[token]
            parent = self.token_parents[token]
            if birth < 0:
                if target is not None or parent is not None:
                    raise ValueError(f"inactive token {token} contains clone data")
                continue
            if target is None or not 0 <= target < self.state_count:
                raise ValueError(f"active token {token} has an invalid target")
            if token < self.base_token_count:
                if birth != 0 or parent is not None:
                    raise ValueError(f"base token {token} is not canonical")
            elif parent is None or not 0 <= parent < token:
                raise ValueError(f"merged token {token} has an invalid clone parent")
            elif birth <= self.token_births[parent]:
                raise ValueError(f"merged token {token} does not postdate its parent")
        for state, excluded in enumerate(self.state_exclusions):
            if any(not 0 <= token < self.alphabet_size for token in excluded):
                raise ValueError(f"state {state} excludes an out-of-range token")

    @property
    def state_count(self) -> int:
        return len(self.state_parents)

    @property
    def active_tokens(self) -> tuple[int, ...]:
        return tuple(
            token for token, birth in enumerate(self.token_births) if birth >= 0
        )

    @property
    def dense_cell_count(self) -> int:
        return self.state_count * len(self.active_tokens)

    def _check_state(self, state: int) -> None:
        if not 0 <= state < self.state_count:
            raise ValueError("persistent adjacency state is out of range")

    def _check_token(self, token: int) -> None:
        if not 0 <= token < self.alphabet_size:
            raise ValueError("persistent adjacency token is out of range")

    def allowed(self, state: int, token: int) -> bool:
        """Query one logical DFA cell without materializing a transition row."""
        self._check_state(state)
        self._check_token(token)
        key = (state, token)
        known = self._allowed_cache.get(key)
        if known is not None:
            return known
        if self.token_births[token] < 0:
            self._allowed_cache[key] = False
            return False

        query_state = state
        query_token = token
        trail: list[tuple[int, int]] = []
        while True:
            cached = self._allowed_cache.get((query_state, query_token))
            if cached is not None:
                result = cached
                break
            trail.append((query_state, query_token))
            if self.state_births[query_state] > self.token_births[query_token]:
                if query_token in self.state_exclusions[query_state]:
                    result = False
                    break
                parent_state = self.state_parents[query_state]
                if parent_state is None:
                    result = True
                    break
                query_state = parent_state
                continue
            parent_token = self.token_parents[query_token]
            if parent_token is None:
                result = True
                break
            query_token = parent_token
        for cell in trail:
            self._allowed_cache[cell] = result
        return result

    def transition(self, state: int, token: int) -> int | None:
        """Return the 1-local target iff the persistent cell is allowed."""
        if not self.allowed(state, token):
            return None
        target = self.token_targets[token]
        if target is None:
            raise AssertionError("an allowed token must have a target")
        return target

    def transition_groups(self, state: int) -> tuple[Transition, ...]:
        """Materialize one row, grouped by its 1-local target."""
        self._check_state(state)
        target_bits: dict[int, int] = {}
        for token in self.active_tokens:
            target = self.transition(state, token)
            if target is not None:
                target_bits[target] = target_bits.get(target, 0) | (1 << token)
        return tuple(
            Transition(SymbolSet(self.alphabet_size, bits), target)
            for target, bits in sorted(target_bits.items())
        )

    def iter_allowed_tokens(self, state: int) -> Iterator[int]:
        self._check_state(state)
        return (token for token in self.active_tokens if self.allowed(state, token))

    def to_dfa(self, *, max_cells: int | None = 10_000_000) -> DFA[bool]:
        """Materialize the equivalent DFA for testing or small prefixes."""
        if max_cells is not None and self.dense_cell_count > max_cells:
            raise RuntimeError(
                "persistent adjacency materialization budget exceeded: "
                f"{self.dense_cell_count} cells"
            )
        rows = tuple(self.transition_groups(state) for state in range(self.state_count))
        return DFA.accepting(
            self.alphabet_size,
            0,
            range(self.state_count),
            rows,
        )


@dataclass(frozen=True, slots=True)
class CanonicalAdjacencyResult:
    adjacency: CanonicalAdjacencyIR
    metrics: CanonicalAdjacencyMetrics


type CanonicalAdjacencyProgress = Callable[[CanonicalAdjacencyMetrics], None]


class CanonicalAdjacencyCompiler:
    """Apply BPE merges directly to the persistent 1-local representation."""

    def __init__(
        self,
        tokens: Sequence[bytes | None],
        parents: Any,
        *,
        base_token_count: int = 256,
    ) -> None:
        self.tokens = tuple(tokens)
        self.parents = parents
        self.base_token_count = base_token_count
        if len(parents) != len(self.tokens):
            raise ValueError("tokens and parents must have the same length")
        if not 0 < base_token_count <= len(self.tokens):
            raise ValueError("invalid base token count")
        base_tokens = self.tokens[:base_token_count]
        if any(token is None or len(token) != 1 for token in base_tokens):
            raise ValueError("every base token must contain exactly one byte")
        if len(set(base_tokens)) != len(base_tokens):
            raise ValueError("base tokens must be unique")

    def _metrics(
        self,
        adjacency: CanonicalAdjacencyIR,
        started: float,
    ) -> CanonicalAdjacencyMetrics:
        active_count = len(adjacency.active_tokens)
        state_links = adjacency.state_count - 1
        token_links = sum(parent is not None for parent in adjacency.token_parents)
        exclusions = sum(map(len, adjacency.state_exclusions))
        return CanonicalAdjacencyMetrics(
            adjacency.applied_merges,
            adjacency.state_count,
            active_count,
            state_links,
            token_links,
            exclusions,
            adjacency.state_count * active_count,
            adjacency.state_count
            + active_count
            + state_links
            + token_links
            + exclusions,
            time.perf_counter() - started,
        )

    def compile(
        self,
        *,
        merge_limit: int | None = None,
        checkpoint_interval: int = 1_000,
        progress: CanonicalAdjacencyProgress | None = None,
    ) -> CanonicalAdjacencyResult:
        if merge_limit is not None and merge_limit < 0:
            raise ValueError("merge limit must be non-negative")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        started = time.perf_counter()
        token_parents: list[int | None] = [None] * len(self.tokens)
        token_births = [-1] * len(self.tokens)
        token_targets: list[int | None] = [None] * len(self.tokens)
        for token in range(self.base_token_count):
            token_births[token] = 0
            token_targets[token] = 0
        state_parents: list[int | None] = [None]
        state_births = [0]
        state_exclusions: list[frozenset[int]] = [frozenset()]
        applied = 0

        def snapshot() -> CanonicalAdjacencyIR:
            return CanonicalAdjacencyIR(
                len(self.tokens),
                self.base_token_count,
                tuple(state_parents),
                tuple(state_births),
                tuple(state_exclusions),
                tuple(token_parents),
                tuple(token_births),
                tuple(token_targets),
                applied,
            )

        for child in range(self.base_token_count, len(self.tokens)):
            if self.tokens[child] is None:
                continue
            if merge_limit is not None and applied >= merge_limit:
                break
            parent = self.parents[child]
            if len(parent) != 2:
                raise ValueError(f"rank {child} does not have two parents")
            left, right = (int(value) for value in parent)
            if not (0 <= left < child and 0 <= right < child):
                raise ValueError(
                    f"rank {child} has invalid merge parents {(left, right)}"
                )
            middle = token_targets[left]
            right_target = token_targets[right]
            if middle is None or right_target is None:
                raise ValueError(f"rank {child} refers to an inactive parent")

            token_parents[child] = left
            token_births[child] = 2 * applied + 1
            token_targets[child] = right_target
            fresh = len(state_parents)
            state_parents.append(middle)
            state_births.append(2 * applied + 2)
            denied = {right}
            if left == right:
                denied.add(child)
            state_exclusions.append(frozenset(denied))
            token_targets[left] = fresh
            applied += 1

            if progress is not None and (
                applied % checkpoint_interval == 0
                or child + 1 == len(self.tokens)
                or (merge_limit is not None and applied == merge_limit)
            ):
                adjacency = snapshot()
                progress(self._metrics(adjacency, started))

        adjacency = snapshot()
        return CanonicalAdjacencyResult(
            adjacency,
            self._metrics(adjacency, started),
        )


__all__ = [
    "CanonicalAdjacencyCompiler",
    "CanonicalAdjacencyIR",
    "CanonicalAdjacencyMetrics",
    "CanonicalAdjacencyProgress",
    "CanonicalAdjacencyResult",
]
