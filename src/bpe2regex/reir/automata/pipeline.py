from collections.abc import Iterable
from dataclasses import dataclass

from ..builder import DEFAULT_BUILDER, RegexBuilder
from ..ops import Op
from .absorption import (
    AutomatonSemanticAbsorber,
    SemanticAbsorptionResult,
)
from .algorithms import minimize_dfa
from .defaults import encode_default_transitions
from .elimination import ArdenEliminator
from .ir import DFA


@dataclass(frozen=True, slots=True)
class AutomatonCompilationResult:
    """Observable stages of a pure acceptance-automata compilation."""

    expression: Op
    minimized_automata: tuple[DFA[bool], ...]
    absorption: SemanticAbsorptionResult
    encoded_automata: tuple[DFA[bool], ...]


class AcceptanceAutomataCompiler:
    """Compile a union of pure acceptance DFAs through the automata pipeline."""

    def __init__(
        self,
        eliminator: ArdenEliminator | None = None,
        absorber: AutomatonSemanticAbsorber | None = None,
        *,
        builder: RegexBuilder = DEFAULT_BUILDER,
    ) -> None:
        self.builder = builder
        self.eliminator = (
            ArdenEliminator(builder=builder) if eliminator is None else eliminator
        )
        self.absorber = AutomatonSemanticAbsorber() if absorber is None else absorber

    def run(
        self,
        alternatives: Iterable[DFA[bool]],
    ) -> AutomatonCompilationResult:
        minimized = tuple(
            minimize_dfa(automaton).automaton for automaton in alternatives
        )
        absorption = self.absorber.run(minimized)
        encoded = tuple(
            encode_default_transitions(automaton)
            for automaton in absorption.alternatives
        )
        expression = self.builder.alternate(
            *(self.eliminator.lower(automaton) for automaton in encoded)
        )
        return AutomatonCompilationResult(
            expression,
            minimized,
            absorption,
            encoded,
        )

    def compile(self, alternatives: Iterable[DFA[bool]]) -> Op:
        return self.run(alternatives).expression


__all__ = ["AcceptanceAutomataCompiler", "AutomatonCompilationResult"]
