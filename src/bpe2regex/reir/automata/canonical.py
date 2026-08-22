import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .ir import DFA, Transition
from .labels import SymbolSet

type _TransitionRow = dict[int, int]


@dataclass(frozen=True, slots=True)
class CanonicalTokenDFAMetrics:
    applied_merges: int
    state_count: int
    transition_group_count: int
    explicit_transition_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class CanonicalTokenDFAResult:
    automaton: DFA[bool]
    metrics: CanonicalTokenDFAMetrics


class CanonicalTokenDFABudgetExceeded(RuntimeError):
    def __init__(self, metrics: CanonicalTokenDFAMetrics, reason: str) -> None:
        super().__init__(f"canonical token DFA budget exceeded: {reason}")
        self.metrics = metrics
        self.reason = reason


type CanonicalTokenDFAProgress = Callable[[CanonicalTokenDFAMetrics], None]


def _target(row: _TransitionRow, symbol: int) -> int | None:
    mask = 1 << symbol
    for target, bits in row.items():
        if bits & mask:
            return target
    return None


def _remove(row: _TransitionRow, symbol: int) -> int:
    mask = 1 << symbol
    for target, bits in tuple(row.items()):
        if not bits & mask:
            continue
        remaining = bits ^ mask
        if remaining:
            row[target] = remaining
            return 0
        del row[target]
        return -1
    return 0


def _set(row: _TransitionRow, symbol: int, target: int) -> int:
    previous = _target(row, symbol)
    if previous == target:
        return 0
    delta = _remove(row, symbol)
    if target in row:
        row[target] |= 1 << symbol
        return delta
    row[target] = 1 << symbol
    return delta + 1


class CanonicalTokenDFACompiler:
    """Construct the universal canonical-token DFA by applying BPE merges."""

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
        rows: list[_TransitionRow],
        applied_merges: int,
        transition_groups: int,
        started: float,
    ) -> CanonicalTokenDFAMetrics:
        return CanonicalTokenDFAMetrics(
            applied_merges,
            len(rows),
            transition_groups,
            sum(bits.bit_count() for row in rows for bits in row.values()),
            time.perf_counter() - started,
        )

    def compile(
        self,
        *,
        merge_limit: int | None = None,
        max_states: int | None = None,
        max_transition_groups: int | None = None,
        checkpoint_interval: int = 1_000,
        progress: CanonicalTokenDFAProgress | None = None,
    ) -> CanonicalTokenDFAResult:
        if merge_limit is not None and merge_limit < 0:
            raise ValueError("merge limit must be non-negative")
        if max_states is not None and max_states <= 0:
            raise ValueError("state budget must be positive")
        if max_transition_groups is not None and max_transition_groups <= 0:
            raise ValueError("transition-group budget must be positive")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint interval must be positive")

        started = time.perf_counter()
        base_bits = (1 << self.base_token_count) - 1
        rows: list[_TransitionRow] = [{0: base_bits}]
        transition_groups = 1
        applied = 0

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

            triples: list[tuple[int, int, int]] = []
            for source, row in enumerate(rows):
                middle = _target(row, left)
                if middle is None:
                    continue
                target = _target(rows[middle], right)
                if target is not None:
                    triples.append((source, middle, target))
            middle_states = tuple(sorted({middle for _, middle, _ in triples}))
            if not triples:
                raise ValueError(
                    f"rank {child} merge {(left, right)} has no universal-DFA context"
                )
            if max_states is not None and len(rows) + len(middle_states) > max_states:
                metrics = self._metrics(rows, applied, transition_groups, started)
                raise CanonicalTokenDFABudgetExceeded(metrics, "state count")

            for source, _, target in triples:
                transition_groups += _set(rows[source], child, target)

            fresh: dict[int, int] = {}
            for middle in middle_states:
                copied = rows[middle].copy()
                transition_groups += len(copied)
                transition_groups += _remove(copied, right)
                if left == right:
                    transition_groups += _remove(copied, child)
                fresh[middle] = len(rows)
                rows.append(copied)

            for row in rows:
                target = _target(row, left)
                replacement = None if target is None else fresh.get(target)
                if replacement is not None:
                    transition_groups += _set(row, left, replacement)

            applied += 1
            if (
                max_transition_groups is not None
                and transition_groups > max_transition_groups
            ):
                metrics = self._metrics(rows, applied, transition_groups, started)
                raise CanonicalTokenDFABudgetExceeded(metrics, "transition-group count")
            if progress is not None and (
                applied % checkpoint_interval == 0
                or child + 1 == len(self.tokens)
                or (merge_limit is not None and applied == merge_limit)
            ):
                progress(self._metrics(rows, applied, transition_groups, started))

        automaton = DFA.accepting(
            len(self.tokens),
            0,
            range(len(rows)),
            tuple(
                tuple(
                    Transition(SymbolSet(len(self.tokens), bits), target)
                    for target, bits in row.items()
                )
                for row in rows
            ),
        )
        return CanonicalTokenDFAResult(
            automaton,
            self._metrics(rows, applied, transition_groups, started),
        )


__all__ = [
    "CanonicalTokenDFABudgetExceeded",
    "CanonicalTokenDFACompiler",
    "CanonicalTokenDFAMetrics",
    "CanonicalTokenDFAProgress",
    "CanonicalTokenDFAResult",
]
