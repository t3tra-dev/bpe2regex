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
class DefaultTransition:
    """A fallback target for symbols not covered by explicit transitions."""

    target: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, int):
            raise TypeError("an automaton default target must be an integer")
        if self.target < 0:
            raise ValueError("an automaton default target must be non-negative")


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
    defaults: tuple[DefaultTransition | None, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if self.alphabet_size <= 0:
            raise ValueError("a DFA alphabet must contain at least one symbol")
        if not self.outputs:
            raise ValueError("a DFA must contain at least one state")
        if len(self.transitions) != len(self.outputs):
            raise ValueError("DFA outputs and transition rows must have equal length")
        defaults = (
            (None,) * len(self.outputs) if not self.defaults else tuple(self.defaults)
        )
        if len(defaults) != len(self.outputs):
            raise ValueError("DFA outputs and default rows must have equal length")
        if not 0 <= self.start < len(self.outputs):
            raise ValueError("the DFA start state is out of range")
        for output in self.outputs:
            if output is not None:
                try:
                    hash(output)
                except TypeError as error:
                    raise TypeError("a DFA output must be hashable") from error

        canonical_rows: list[tuple[Transition, ...]] = []
        canonical_defaults: list[DefaultTransition | None] = []
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
            default = defaults[state]
            if default is not None:
                if not isinstance(default, DefaultTransition):
                    raise TypeError("a DFA default row must contain DefaultTransition")
                if default.target >= len(self.outputs):
                    raise ValueError(
                        f"state {state} has an out-of-range default target"
                    )
                covered = 0
                for transition in canonical_rows[-1]:
                    covered |= transition.symbols.bits
                if covered == (1 << self.alphabet_size) - 1:
                    raise ValueError(
                        f"state {state} has a fully shadowed default transition"
                    )
            canonical_defaults.append(default)
        object.__setattr__(self, "transitions", tuple(canonical_rows))
        object.__setattr__(self, "defaults", tuple(canonical_defaults))

    @classmethod
    def accepting(
        cls,
        alphabet_size: int,
        start: int,
        accepting_states: Iterable[int],
        transitions: tuple[tuple[Transition, ...], ...],
        defaults: tuple[DefaultTransition | None, ...] = (),
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
            defaults,
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
        return sum(map(len, self.transitions)) + self.default_transition_count

    @property
    def default_transition_count(self) -> int:
        return sum(default is not None for default in self.defaults)

    @property
    def explicit_symbol_transition_count(self) -> int:
        return sum(
            len(transition.symbols) for row in self.transitions for transition in row
        )

    @property
    def is_total(self) -> bool:
        full_bits = (1 << self.alphabet_size) - 1
        for row, default in zip(self.transitions, self.defaults, strict=True):
            if default is not None:
                continue
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
        default = self.defaults[state]
        return None if default is None else default.target

    def effective_transitions(self, state: int) -> tuple[Transition, ...]:
        """Expand one default edge to its exact complement symbol set."""
        self._check_state(state)
        row = self.transitions[state]
        default = self.defaults[state]
        if default is None:
            return row
        covered = SymbolSet.empty(self.alphabet_size)
        target_bits: dict[int, int] = {}
        for transition in row:
            covered |= transition.symbols
            target_bits[transition.target] = (
                target_bits.get(transition.target, 0) | transition.symbols.bits
            )
        remaining = SymbolSet.full(self.alphabet_size) - covered
        target_bits[default.target] = (
            target_bits.get(default.target, 0) | remaining.bits
        )
        return tuple(
            sorted(
                (
                    Transition(SymbolSet(self.alphabet_size, bits), target)
                    for target, bits in target_bits.items()
                ),
                key=lambda transition: (
                    (transition.symbols.bits & -transition.symbols.bits).bit_length(),
                    transition.target,
                ),
            )
        )

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
        rows: list[tuple[Transition, ...]] = []
        defaults: list[DefaultTransition | None] = []
        for row, default in zip(self.transitions, self.defaults, strict=True):
            rows.append(row)
            covered = 0
            for transition in row:
                covered |= transition.symbols.bits
            defaults.append(
                default
                if default is not None or covered == (1 << self.alphabet_size) - 1
                else DefaultTransition(sink)
            )
        rows.append(())
        defaults.append(DefaultTransition(sink))
        return DFA(
            self.alphabet_size,
            self.start,
            (*self.outputs, None),
            tuple(rows),
            tuple(defaults),
        )


__all__ = ["DFA", "DefaultTransition", "Transition"]
