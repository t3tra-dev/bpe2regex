from abc import ABC, abstractmethod
from collections.abc import Hashable

from .ir import DFA, DefaultTransition, Transition


class DefaultTargetSelector(ABC):
    """Extensible policy for selecting one dense transition as a default."""

    @abstractmethod
    def select(self, transitions: tuple[Transition, ...]) -> int | None:
        """Return a transition target, or ``None`` to keep an explicit row."""


class LargestSymbolSetDefault(DefaultTargetSelector):
    """Select the target covering most symbols, with a stable target-id tie break."""

    def select(self, transitions: tuple[Transition, ...]) -> int | None:
        if not transitions:
            return None
        return min(
            transitions,
            key=lambda transition: (-len(transition.symbols), transition.target),
        ).target


def encode_default_transitions[OutputT: Hashable](
    automaton: DFA[OutputT],
    selector: DefaultTargetSelector | None = None,
) -> DFA[OutputT]:
    """Use one fallback target for every total row where it saves a label."""
    policy = LargestSymbolSetDefault() if selector is None else selector
    rows: list[tuple[Transition, ...]] = []
    defaults: list[DefaultTransition | None] = []
    for state in range(automaton.state_count):
        effective = automaton.effective_transitions(state)
        covered = sum(len(transition.symbols) for transition in effective)
        if covered != automaton.alphabet_size:
            rows.append(effective)
            defaults.append(None)
            continue
        target = policy.select(effective)
        if target is None:
            rows.append(effective)
            defaults.append(None)
            continue
        rows.append(
            tuple(transition for transition in effective if transition.target != target)
        )
        defaults.append(DefaultTransition(target))
    return DFA(
        automaton.alphabet_size,
        automaton.start,
        automaton.outputs,
        tuple(rows),
        tuple(defaults),
    )


def expand_default_transitions[OutputT: Hashable](
    automaton: DFA[OutputT],
) -> DFA[OutputT]:
    """Lower default syntax back to explicit disjoint symbol-set edges."""
    return DFA(
        automaton.alphabet_size,
        automaton.start,
        automaton.outputs,
        tuple(
            automaton.effective_transitions(state)
            for state in range(automaton.state_count)
        ),
    )


__all__ = [
    "DefaultTargetSelector",
    "LargestSymbolSetDefault",
    "encode_default_transitions",
    "expand_default_transitions",
]
