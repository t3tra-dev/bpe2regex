from collections import deque
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from itertools import chain

from .ir import DFA, DefaultTransition, Transition
from .labels import SymbolSet

type ProductState = tuple[int | None, int | None]


@dataclass(frozen=True, slots=True)
class AutomatonTransform[OutputT: Hashable]:
    """A transformed automaton and the source-to-result state quotient map."""

    automaton: DFA[OutputT]
    state_map: tuple[int | None, ...]
    blocks: tuple[frozenset[int], ...]


def _symbol_partition(
    alphabet_size: int,
    labels: Iterable[SymbolSet],
) -> tuple[SymbolSet, ...]:
    blocks = [SymbolSet.full(alphabet_size).bits]
    for label in labels:
        if label.alphabet_size != alphabet_size:
            raise ValueError("cannot partition labels from different alphabets")
        refined: list[int] = []
        changed = False
        for block in blocks:
            inside = block & label.bits
            outside = block & ~label.bits
            if inside and outside:
                refined.extend((inside, outside))
                changed = True
            else:
                refined.append(block)
        if changed:
            blocks = refined
    return tuple(
        SymbolSet(alphabet_size, bits)
        for bits in sorted(blocks, key=lambda bits: (bits & -bits).bit_length())
    )


def alphabet_partition[OutputT: Hashable](
    automaton: DFA[OutputT],
) -> tuple[SymbolSet, ...]:
    """Group symbols whose transition behavior is indistinguishable everywhere."""
    return _symbol_partition(
        automaton.alphabet_size,
        (
            transition.symbols
            for state in range(automaton.state_count)
            for transition in automaton.effective_transitions(state)
        ),
    )


def reachable_states[OutputT: Hashable](automaton: DFA[OutputT]) -> tuple[int, ...]:
    """Return states in deterministic breadth-first discovery order."""
    order: list[int] = []
    pending = deque((automaton.start,))
    seen = {automaton.start}
    while pending:
        state = pending.popleft()
        order.append(state)
        for transition in automaton.effective_transitions(state):
            if transition.target not in seen:
                seen.add(transition.target)
                pending.append(transition.target)
    return tuple(order)


def coreachable_states[OutputT: Hashable](automaton: DFA[OutputT]) -> frozenset[int]:
    """Return states from which some observable accepting output is reachable."""
    predecessors: list[set[int]] = [set() for _ in range(automaton.state_count)]
    for source in range(automaton.state_count):
        for transition in automaton.effective_transitions(source):
            predecessors[transition.target].add(source)
    coreachable = set(automaton.accepting_states)
    pending = deque(sorted(coreachable))
    while pending:
        target = pending.popleft()
        for source in sorted(predecessors[target]):
            if source not in coreachable:
                coreachable.add(source)
                pending.append(source)
    return frozenset(coreachable)


def prune_unreachable[OutputT: Hashable](
    automaton: DFA[OutputT],
) -> AutomatonTransform[OutputT]:
    order = reachable_states(automaton)
    state_map: list[int | None] = [None] * automaton.state_count
    for target, source in enumerate(order):
        state_map[source] = target
    rows: list[tuple[Transition, ...]] = []
    defaults: list[DefaultTransition | None] = []
    for source in order:
        rewritten: list[Transition] = []
        for transition in automaton.transitions[source]:
            target = state_map[transition.target]
            if target is None:
                raise AssertionError(
                    "a reachable transition must have a reachable target"
                )
            rewritten.append(Transition(transition.symbols, target))
        rows.append(tuple(rewritten))
        default = automaton.defaults[source]
        if default is None:
            defaults.append(None)
        else:
            target = state_map[default.target]
            if target is None:
                raise AssertionError(
                    "a reachable default transition must have a reachable target"
                )
            defaults.append(DefaultTransition(target))
    result = DFA(
        automaton.alphabet_size,
        0,
        tuple(automaton.outputs[source] for source in order),
        tuple(rows),
        tuple(defaults),
    )
    return AutomatonTransform(
        result,
        tuple(state_map),
        tuple(frozenset((source,)) for source in order),
    )


def _canonical_reindex[OutputT: Hashable](
    automaton: DFA[OutputT],
) -> AutomatonTransform[OutputT]:
    return prune_unreachable(automaton)


def minimize_dfa[OutputT: Hashable](
    automaton: DFA[OutputT],
) -> AutomatonTransform[OutputT]:
    """Compute the output-aware residual quotient of a partial DFA.

    Missing transitions are modeled by a rejecting sink, so the result is a
    canonical, reachable, total DFA.
    """
    pruned = prune_unreachable(automaton)
    complete = pruned.automaton.totalize()
    symbol_classes = alphabet_partition(complete)
    representatives = tuple(
        symbol_class.first_symbol for symbol_class in symbol_classes
    )
    if any(symbol is None for symbol in representatives):
        raise AssertionError("an alphabet partition must not contain empty classes")
    transition_table = tuple(
        tuple(
            complete.transition(state, symbol)
            for symbol in representatives
            if symbol is not None
        )
        for state in range(complete.state_count)
    )
    if any(target is None for row in transition_table for target in row):
        raise AssertionError("a total DFA must define every transition")

    output_groups: dict[OutputT | None, set[int]] = {}
    for state, output in enumerate(complete.outputs):
        output_groups.setdefault(output, set()).add(state)
    blocks: set[frozenset[int]] = {frozenset(group) for group in output_groups.values()}
    work = deque(sorted(blocks, key=lambda block: (len(block), min(block))))
    pending = set(blocks)

    predecessors: list[dict[int, set[int]]] = [{} for _ in range(len(symbol_classes))]
    for source, row in enumerate(transition_table):
        for symbol_class, target in enumerate(row):
            assert target is not None
            predecessors[symbol_class].setdefault(target, set()).add(source)

    def enqueue(block: frozenset[int]) -> None:
        if block not in pending:
            pending.add(block)
            work.append(block)

    while work:
        splitter = work.popleft()
        if splitter not in pending:
            continue
        pending.remove(splitter)
        for symbol_class in range(len(symbol_classes)):
            sources: set[int] = set()
            reverse = predecessors[symbol_class]
            for target in splitter:
                sources.update(reverse.get(target, ()))
            if not sources:
                continue
            for block in sorted(blocks, key=lambda item: (len(item), min(item))):
                inside = frozenset(block & sources)
                outside = block - sources
                if not inside or not outside:
                    continue
                blocks.remove(block)
                blocks.update((inside, outside))
                if block in pending:
                    pending.remove(block)
                    enqueue(inside)
                    enqueue(outside)
                else:
                    enqueue(
                        inside
                        if (len(inside), min(inside)) <= (len(outside), min(outside))
                        else outside
                    )

    ordered_blocks = tuple(sorted(blocks, key=lambda block: (min(block), len(block))))
    complete_to_block = [0] * complete.state_count
    for block_index, block in enumerate(ordered_blocks):
        for state in block:
            complete_to_block[state] = block_index

    quotient_rows: list[tuple[Transition, ...]] = []
    quotient_outputs: list[OutputT | None] = []
    for block in ordered_blocks:
        representative = min(block)
        quotient_outputs.append(complete.outputs[representative])
        target_bits: dict[int, int] = {}
        for symbol_class, target in zip(
            symbol_classes,
            transition_table[representative],
            strict=True,
        ):
            assert target is not None
            quotient_target = complete_to_block[target]
            target_bits[quotient_target] = (
                target_bits.get(quotient_target, 0) | symbol_class.bits
            )
        quotient_rows.append(
            tuple(
                Transition(
                    SymbolSet(complete.alphabet_size, bits),
                    target,
                )
                for target, bits in target_bits.items()
            )
        )

    quotient = DFA(
        complete.alphabet_size,
        complete_to_block[complete.start],
        tuple(quotient_outputs),
        tuple(quotient_rows),
    )
    canonical = _canonical_reindex(quotient)

    source_to_target: list[int | None] = [None] * automaton.state_count
    result_blocks: list[set[int]] = [
        set() for _ in range(canonical.automaton.state_count)
    ]
    for source, pruned_state in enumerate(pruned.state_map):
        if pruned_state is None:
            continue
        quotient_state = complete_to_block[pruned_state]
        target = canonical.state_map[quotient_state]
        assert target is not None
        source_to_target[source] = target
        result_blocks[target].add(source)
    return AutomatonTransform(
        canonical.automaton,
        tuple(source_to_target),
        tuple(frozenset(block) for block in result_blocks),
    )


def combined_alphabet_partition[
    LeftOutputT: Hashable,
    RightOutputT: Hashable,
](
    left: DFA[LeftOutputT],
    right: DFA[RightOutputT],
) -> tuple[SymbolSet, ...]:
    """Partition a shared alphabet by both automata's transition behavior."""
    if left.alphabet_size != right.alphabet_size:
        raise ValueError("cannot compare DFAs over different alphabets")
    return _symbol_partition(
        left.alphabet_size,
        chain(
            (
                transition.symbols
                for state in range(left.state_count)
                for transition in left.effective_transitions(state)
            ),
            (
                transition.symbols
                for state in range(right.state_count)
                for transition in right.effective_transitions(state)
            ),
        ),
    )


def equivalence_counterexample[
    LeftOutputT: Hashable,
    RightOutputT: Hashable,
](
    left: DFA[LeftOutputT],
    right: DFA[RightOutputT],
) -> tuple[int, ...] | None:
    """Return the shortest lexicographic word with differing output, if any."""
    symbol_classes = combined_alphabet_partition(left, right)
    symbols = tuple(symbol_class.first_symbol for symbol_class in symbol_classes)
    if any(symbol is None for symbol in symbols):
        raise AssertionError("an alphabet partition must not contain empty classes")

    start: ProductState = left.start, right.start
    pending: deque[ProductState] = deque((start,))
    parents: dict[ProductState, tuple[ProductState, int] | None] = {start: None}

    def word(state: ProductState) -> tuple[int, ...]:
        result: list[int] = []
        current = state
        parent = parents[current]
        while parent is not None:
            previous, symbol = parent
            result.append(symbol)
            current = previous
            parent = parents[current]
        result.reverse()
        return tuple(result)

    while pending:
        state = pending.popleft()
        left_output = None if state[0] is None else left.outputs[state[0]]
        right_output = None if state[1] is None else right.outputs[state[1]]
        if left_output != right_output:
            return word(state)
        for symbol in symbols:
            assert symbol is not None
            successor = (
                None if state[0] is None else left.transition(state[0], symbol),
                None if state[1] is None else right.transition(state[1], symbol),
            )
            if successor not in parents:
                parents[successor] = state, symbol
                pending.append(successor)
    return None


def equivalent[
    LeftOutputT: Hashable,
    RightOutputT: Hashable,
](
    left: DFA[LeftOutputT],
    right: DFA[RightOutputT],
) -> bool:
    return equivalence_counterexample(left, right) is None


__all__ = [
    "AutomatonTransform",
    "alphabet_partition",
    "combined_alphabet_partition",
    "coreachable_states",
    "equivalence_counterexample",
    "equivalent",
    "minimize_dfa",
    "prune_unreachable",
    "reachable_states",
]
