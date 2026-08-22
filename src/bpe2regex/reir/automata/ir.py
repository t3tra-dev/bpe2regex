from collections.abc import Hashable, Iterable
from dataclasses import dataclass

from .labels import SymbolSet


@dataclass(frozen=True, slots=True)
class Transition:
    """One non-empty symbol partition leading to a deterministic target."""

    symbols: SymbolSet
    target: int

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("an automaton transition must consume at least one symbol")
        if not isinstance(self.target, int):
            raise TypeError("an automaton transition target must be an integer")
        if self.target < 0:
            raise ValueError("an automaton transition target must be non-negative")


@dataclass(frozen=True, slots=True)
class DFA[OutputT: Hashable]:
    """Immutable partial DFA with optional accepting-state outputs.

    ``None`` denotes rejection. Any other hashable value is observable output,
    so minimization is safe for both pure acceptance and tagged rank semantics.
    """

    alphabet_size: int
    start: int
    outputs: tuple[OutputT | None, ...]
    transitions: tuple[tuple[Transition, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if self.alphabet_size <= 0:
            raise ValueError("a DFA alphabet must contain at least one symbol")
        if not self.outputs:
            raise ValueError("a DFA must contain at least one state")
        if len(self.transitions) != len(self.outputs):
            raise ValueError("DFA outputs and transition rows must have equal length")
        if not 0 <= self.start < len(self.outputs):
            raise ValueError("the DFA start state is out of range")
        for output in self.outputs:
            if output is not None:
                try:
                    hash(output)
                except TypeError as error:
                    raise TypeError("a DFA output must be hashable") from error

        canonical_rows: list[tuple[Transition, ...]] = []
        for state, row in enumerate(self.transitions):
            target_bits: dict[int, int] = {}
            for transition in row:
                if not isinstance(transition, Transition):
                    raise TypeError("a DFA transition row must contain Transitions")
                if transition.symbols.alphabet_size != self.alphabet_size:
                    raise ValueError(
                        f"state {state} contains a transition from another alphabet"
                    )
                if transition.target >= len(self.outputs):
                    raise ValueError(
                        f"state {state} has an out-of-range transition target"
                    )
                for target, bits in target_bits.items():
                    if target != transition.target and bits & transition.symbols.bits:
                        raise ValueError(
                            f"state {state} has overlapping deterministic transitions"
                        )
                target_bits[transition.target] = (
                    target_bits.get(transition.target, 0) | transition.symbols.bits
                )
            canonical_rows.append(
                tuple(
                    sorted(
                        (
                            Transition(SymbolSet(self.alphabet_size, bits), target)
                            for target, bits in target_bits.items()
                        ),
                        key=lambda item: (
                            (item.symbols.bits & -item.symbols.bits).bit_length(),
                            item.target,
                            item.symbols.bits,
                        ),
                    )
                )
            )
        object.__setattr__(self, "transitions", tuple(canonical_rows))

    @classmethod
    def accepting(
        cls,
        alphabet_size: int,
        start: int,
        accepting_states: Iterable[int],
        transitions: tuple[tuple[Transition, ...], ...],
    ) -> DFA[bool]:
        accepting = frozenset(accepting_states)
        if any(not 0 <= state < len(transitions) for state in accepting):
            raise ValueError("an accepting state is out of range")
        return DFA(
            alphabet_size,
            start,
            tuple(
                True if state in accepting else None
                for state in range(len(transitions))
            ),
            transitions,
        )

    @property
    def state_count(self) -> int:
        return len(self.outputs)

    @property
    def accepting_states(self) -> frozenset[int]:
        return frozenset(
            state for state, output in enumerate(self.outputs) if output is not None
        )

    @property
    def transition_group_count(self) -> int:
        return sum(map(len, self.transitions))

    @property
    def explicit_symbol_transition_count(self) -> int:
        return sum(
            len(transition.symbols) for row in self.transitions for transition in row
        )

    @property
    def is_total(self) -> bool:
        full_bits = (1 << self.alphabet_size) - 1
        for row in self.transitions:
            covered = 0
            for transition in row:
                covered |= transition.symbols.bits
            if covered != full_bits:
                return False
        return True

    def _check_state(self, state: int) -> None:
        if not isinstance(state, int):
            raise TypeError("a DFA state must be an integer")
        if not 0 <= state < self.state_count:
            raise ValueError("a DFA state is out of range")

    def _check_symbol(self, symbol: int) -> None:
        if not isinstance(symbol, int):
            raise TypeError("a DFA input symbol must be an integer")
        if not 0 <= symbol < self.alphabet_size:
            raise ValueError("a DFA input symbol is outside its alphabet")

    def transition(self, state: int, symbol: int) -> int | None:
        self._check_state(state)
        self._check_symbol(symbol)
        for transition in self.transitions[state]:
            if symbol in transition.symbols:
                return transition.target
        return None

    def run_state(self, value: Iterable[int]) -> int | None:
        state = self.start
        for symbol in value:
            target = self.transition(state, symbol)
            if target is None:
                return None
            state = target
        return state

    def match_output(self, value: Iterable[int]) -> OutputT | None:
        state = self.run_state(value)
        return None if state is None else self.outputs[state]

    def accepts(self, value: Iterable[int]) -> bool:
        return self.match_output(value) is not None

    def trace(self, value: Iterable[int]) -> tuple[int, ...] | None:
        states = [self.start]
        state = self.start
        for symbol in value:
            target = self.transition(state, symbol)
            if target is None:
                return None
            states.append(target)
            state = target
        return tuple(states)

    def totalize(self) -> DFA[OutputT]:
        """Add one rejecting sink when any transition row is partial."""
        if self.is_total:
            return self
        sink = self.state_count
        full = SymbolSet.full(self.alphabet_size)
        rows: list[tuple[Transition, ...]] = []
        for row in self.transitions:
            covered = SymbolSet.empty(self.alphabet_size)
            for transition in row:
                covered |= transition.symbols
            missing = full - covered
            rows.append(row + ((Transition(missing, sink),) if missing else ()))
        rows.append((Transition(full, sink),))
        return DFA(
            self.alphabet_size,
            self.start,
            (*self.outputs, None),
            tuple(rows),
        )


__all__ = ["DFA", "Transition"]
