from abc import ABC, abstractmethod
from collections.abc import Iterable

from .analysis import AnalysisManager, DataFlowAnalysis
from .ops import (
    EPSILON,
    NEVER,
    Alternate,
    CharSet,
    Concat,
    Epsilon,
    Literal,
    Never,
    Op,
    Repeat,
    structural_key,
)


class PatternRewriter:
    """The only API through which a rewrite pattern replaces its root op."""

    def __init__(self, root: Op, analyses: AnalysisManager) -> None:
        self.root = root
        self.analyses = analyses
        self._replacement: Op | None = None

    def replace_op(self, op: Op, replacement: Op) -> None:
        if op is not self.root:
            raise ValueError("a pattern may only replace its current root operation")
        if self._replacement is not None:
            raise ValueError("a pattern may replace its root at most once")
        replacement.verify()
        self._replacement = replacement

    def get_analysis[FactT](
        self,
        analysis_type: type[DataFlowAnalysis[FactT]],
        op: Op | None = None,
    ) -> FactT:
        return self.analyses.get(analysis_type, self.root if op is None else op)

    def take_replacement(self) -> Op | None:
        replacement = self._replacement
        self._replacement = None
        return replacement


class RewritePattern(ABC):
    """Base class for declarative, benefit-ordered operation rewrites."""

    root_type: type[Op] | tuple[type[Op], ...] = Op
    benefit: int = 1

    @abstractmethod
    def match_and_rewrite(self, op: Op, rewriter: PatternRewriter) -> bool:
        """Return true after replacing ``op`` through ``rewriter``."""


class PatternApplicator:
    """Select and apply rewrite patterns to one operation at a time."""

    def __init__(self, patterns: Iterable[RewritePattern]) -> None:
        self.patterns = tuple(
            sorted(patterns, key=lambda pattern: pattern.benefit, reverse=True)
        )

    def apply_once(self, op: Op, analyses: AnalysisManager) -> Op | None:
        for pattern in self.patterns:
            if not isinstance(op, pattern.root_type):
                continue
            rewriter = PatternRewriter(op, analyses)
            matched = pattern.match_and_rewrite(op, rewriter)
            replacement = rewriter.take_replacement()
            if matched != (replacement is not None):
                raise RuntimeError(
                    f"{type(pattern).__name__} returned an inconsistent rewrite result"
                )
            if replacement is None:
                continue
            if replacement == op:
                raise RuntimeError(
                    f"{type(pattern).__name__} replaced an operation with itself"
                )
            return replacement
        return None

    def rewrite_root(
        self,
        root: Op,
        analyses: AnalysisManager | None = None,
        *,
        max_iterations: int = 1_000,
    ) -> Op:
        manager = AnalysisManager() if analyses is None else analyses
        current = root
        for _ in range(max_iterations):
            replacement = self.apply_once(current, manager)
            if replacement is None:
                return current
            current = replacement
        raise RuntimeError("REIR root rewrite did not converge")


class GreedyRewriteDriver:
    """Apply patterns bottom-up until the complete operation graph is stable."""

    def __init__(
        self,
        patterns: Iterable[RewritePattern],
        *,
        max_rewrites: int = 100_000,
    ) -> None:
        self.applicator = PatternApplicator(patterns)
        self.max_rewrites = max_rewrites

    def rewrite(
        self,
        root: Op,
        analyses: AnalysisManager | None = None,
    ) -> Op:
        manager = AnalysisManager() if analyses is None else analyses
        memo: dict[int, tuple[Op, Op]] = {}
        active: set[int] = set()
        rewrite_count = 0

        def rewrite_operands(op: Op) -> Op:
            rewritten = tuple(rewrite(operand) for operand in op.operands)
            return op if rewritten == op.operands else op.with_operands(rewritten)

        def rewrite(op: Op) -> Op:
            nonlocal rewrite_count
            identity = id(op)
            known = memo.get(identity)
            if known is not None and known[0] is op:
                return known[1]
            if identity in active:
                raise ValueError("REIR rewrites require an acyclic operation graph")
            active.add(identity)
            try:
                current = rewrite_operands(op)
                while True:
                    replacement = self.applicator.apply_once(current, manager)
                    if replacement is None:
                        memo[identity] = (op, current)
                        return current
                    rewrite_count += 1
                    if rewrite_count > self.max_rewrites:
                        raise RuntimeError("REIR greedy rewrite did not converge")
                    current = rewrite_operands(replacement)
            finally:
                active.remove(identity)

        return rewrite(root)


def _sum_maximum(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


def _fold_repeat(body: Op, minimum: int, maximum: int | None) -> Op:
    if maximum == 0:
        return EPSILON
    if minimum == 1 and maximum == 1:
        return body
    if isinstance(body, Epsilon):
        return EPSILON
    if isinstance(body, Never):
        return EPSILON if minimum == 0 else NEVER
    if isinstance(body, Repeat) and body.max is None and maximum is None:
        if minimum == 0 and body.min in (0, 1):
            return Repeat(body.body, 0, None)
        if minimum == 1 and body.min in (0, 1):
            return Repeat(body.body, body.min, None)
    return Repeat(body, minimum, maximum)


def _primitive_literal(value: bytes) -> tuple[Literal, int]:
    for width in range(1, len(value) + 1):
        if len(value) % width == 0:
            repetitions = len(value) // width
            primitive = value[:width]
            if primitive * repetitions == value:
                return Literal(primitive), repetitions
    raise AssertionError("every non-empty literal has a primitive root")


def _power_range(op: Op) -> tuple[Op, int, int | None]:
    if isinstance(op, Repeat):
        return op.body, op.min, op.max
    if isinstance(op, Literal):
        body, exponent = _primitive_literal(op.value)
        return body, exponent, exponent
    return op, 1, 1


def _merge_adjacent_powers(parts: list[Op]) -> list[Op]:
    merged: list[Op] = []
    index = 0
    while index < len(parts):
        body, minimum, maximum = _power_range(parts[index])
        end = index + 1
        while end < len(parts):
            next_body, next_minimum, next_maximum = _power_range(parts[end])
            if next_body != body:
                break
            minimum += next_minimum
            maximum = _sum_maximum(maximum, next_maximum)
            end += 1
        if end == index + 1:
            merged.append(parts[index])
        else:
            merged.append(_fold_repeat(body, minimum, maximum))
        index = end
    return merged


def _make_concat(parts: Iterable[Op]) -> Op:
    flattened: list[Op] = []
    pending_literal = bytearray()

    def flush_literal() -> None:
        if pending_literal:
            flattened.append(Literal(bytes(pending_literal)))
            pending_literal.clear()

    pending = list(reversed(tuple(parts)))
    while pending:
        part = pending.pop()
        if isinstance(part, Never):
            return NEVER
        if isinstance(part, Epsilon):
            continue
        if isinstance(part, Concat):
            pending.extend(reversed(part.parts))
        elif isinstance(part, Literal):
            pending_literal.extend(part.value)
        else:
            flush_literal()
            flattened.append(part)
    flush_literal()
    if not flattened:
        return EPSILON
    if len(flattened) == 1:
        return flattened[0]
    return Concat(tuple(flattened))


class NormalizeConcatPattern(RewritePattern):
    root_type = Concat
    benefit = 100

    def match_and_rewrite(self, op: Op, rewriter: PatternRewriter) -> bool:
        assert isinstance(op, Concat)
        replacement = _make_concat(op.parts)
        if isinstance(replacement, Concat):
            merged = _merge_adjacent_powers(list(replacement.parts))
            replacement = _make_concat(merged)
        if replacement == op:
            return False
        rewriter.replace_op(op, replacement)
        return True


class NormalizeAlternatePattern(RewritePattern):
    root_type = Alternate
    benefit = 100

    def match_and_rewrite(self, op: Op, rewriter: PatternRewriter) -> bool:
        assert isinstance(op, Alternate)
        flattened: list[Op] = []
        pending: list[Op] = list(reversed(op.alternatives))
        while pending:
            alternative = pending.pop()
            if isinstance(alternative, Never):
                continue
            if isinstance(alternative, Alternate):
                pending.extend(reversed(alternative.alternatives))
            else:
                flattened.append(alternative)

        flattened = list(dict.fromkeys(flattened))
        atom_indices: list[int] = []
        atom_bits = 0
        for index, alternative in enumerate(flattened):
            if isinstance(alternative, CharSet):
                atom_indices.append(index)
                atom_bits |= alternative.bits
            elif isinstance(alternative, Literal) and len(alternative.value) == 1:
                atom_indices.append(index)
                atom_bits |= 1 << alternative.value[0]
        if len(atom_indices) >= 2:
            atom_index_set = frozenset(atom_indices)
            flattened = [
                alternative
                for index, alternative in enumerate(flattened)
                if index not in atom_index_set
            ]
            flattened.append(CharSet.from_bits(atom_bits))
        flattened.sort(key=structural_key)

        replacement: Op
        if not flattened:
            replacement = NEVER
        elif len(flattened) == 1:
            replacement = flattened[0]
        else:
            replacement = Alternate(tuple(flattened))
        if replacement == op:
            return False
        rewriter.replace_op(op, replacement)
        return True


class NormalizeRepeatPattern(RewritePattern):
    root_type = Repeat
    benefit = 100

    def match_and_rewrite(self, op: Op, rewriter: PatternRewriter) -> bool:
        assert isinstance(op, Repeat)
        replacement = _fold_repeat(op.body, op.min, op.max)
        if replacement == op:
            return False
        rewriter.replace_op(op, replacement)
        return True


def _sequence_units(op: Op) -> tuple[Op, ...]:
    if isinstance(op, Epsilon):
        return ()
    if isinstance(op, Literal):
        return tuple(Literal(bytes((symbol,))) for symbol in op.value)
    if isinstance(op, Concat):
        return tuple(unit for part in op.parts for unit in _sequence_units(part))
    return (op,)


class DiscoverRepeatAlternativesPattern(RewritePattern):
    """Contract a contiguous union of powers into one bounded Repeat."""

    root_type = Alternate
    benefit = 20

    def match_and_rewrite(self, op: Op, rewriter: PatternRewriter) -> bool:
        assert isinstance(op, Alternate)
        base: Op | None = None
        ranges: list[tuple[int, int | None]] = []
        for alternative in op.alternatives:
            if isinstance(alternative, Epsilon):
                ranges.append((0, 0))
                continue
            candidate, minimum, maximum = _power_range(alternative)
            if base is None:
                base = candidate
            elif candidate != base:
                return False
            ranges.append((minimum, maximum))
        if base is None:
            return False

        ranges.sort(key=lambda bounds: bounds[0])
        union_minimum, union_maximum = ranges[0]
        for minimum, maximum in ranges[1:]:
            if union_maximum is None:
                break
            if minimum > union_maximum + 1:
                return False
            union_maximum = None if maximum is None else max(union_maximum, maximum)
        replacement = _fold_repeat(base, union_minimum, union_maximum)
        if replacement == op:
            return False
        rewriter.replace_op(op, replacement)
        return True


class FactorCommonAffixesPattern(RewritePattern):
    """Factor the longest common prefix and suffix of an n-ary union."""

    root_type = Alternate
    benefit = 10

    def match_and_rewrite(self, op: Op, rewriter: PatternRewriter) -> bool:
        assert isinstance(op, Alternate)
        sequences = tuple(_sequence_units(branch) for branch in op.alternatives)
        shortest = min(map(len, sequences))
        prefix_length = 0
        while prefix_length < shortest and all(
            sequence[prefix_length] == sequences[0][prefix_length]
            for sequence in sequences[1:]
        ):
            prefix_length += 1

        suffix_limit = min(len(sequence) - prefix_length for sequence in sequences)
        suffix_length = 0
        while suffix_length < suffix_limit and all(
            sequence[-suffix_length - 1] == sequences[0][-suffix_length - 1]
            for sequence in sequences[1:]
        ):
            suffix_length += 1
        if prefix_length == 0 and suffix_length == 0:
            return False

        middles = []
        for sequence in sequences:
            end = len(sequence) - suffix_length if suffix_length else len(sequence)
            middles.append(_make_concat(sequence[prefix_length:end]))
        middle = Alternate(tuple(middles))
        prefix = sequences[0][:prefix_length]
        suffix = (
            sequences[0][len(sequences[0]) - suffix_length :] if suffix_length else ()
        )
        replacement = _make_concat((*prefix, middle, *suffix))
        rewriter.replace_op(op, replacement)
        return True


CANONICALIZATION_PATTERNS: tuple[RewritePattern, ...] = (
    NormalizeConcatPattern(),
    NormalizeAlternatePattern(),
    NormalizeRepeatPattern(),
)

STRUCTURE_DISCOVERY_PATTERNS: tuple[RewritePattern, ...] = (
    DiscoverRepeatAlternativesPattern(),
    FactorCommonAffixesPattern(),
)

OPTIMIZATION_PATTERNS = (
    *CANONICALIZATION_PATTERNS,
    *STRUCTURE_DISCOVERY_PATTERNS,
)


__all__ = [
    "CANONICALIZATION_PATTERNS",
    "OPTIMIZATION_PATTERNS",
    "STRUCTURE_DISCOVERY_PATTERNS",
    "DiscoverRepeatAlternativesPattern",
    "FactorCommonAffixesPattern",
    "GreedyRewriteDriver",
    "NormalizeAlternatePattern",
    "NormalizeConcatPattern",
    "NormalizeRepeatPattern",
    "PatternApplicator",
    "PatternRewriter",
    "RewritePattern",
]
