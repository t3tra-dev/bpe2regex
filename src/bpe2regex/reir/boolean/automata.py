from collections import deque
from dataclasses import dataclass

from ..automata.ir import DFA, Transition
from ..automata.labels import SymbolSet
from ..ops import BYTE_ALPHABET_SIZE, NEVER, Op, PureOp
from .derivative import BooleanDerivativeEngine


class BooleanDerivativeStateBudgetExceeded(RuntimeError):
    """Raised before derivative closure exceeds its configured state budget."""

    def __init__(self, max_states: int) -> None:
        self.max_states = max_states
        super().__init__(f"Boolean derivative DFA exceeded {max_states} states")


@dataclass(frozen=True, slots=True)
class BooleanDerivativeDFAResult:
    automaton: DFA[bool]
    residuals: tuple[Op, ...]
    derivative_evaluations: int


class BooleanDerivativeDFACompiler:
    """Compile a pure Boolean regex to its finite derivative automaton."""

    def __init__(self, *, max_states: int | None = 10_000) -> None:
        if max_states is not None and max_states <= 0:
            raise ValueError("a derivative DFA state budget must be positive")
        self.max_states = max_states

    def compile(self, root: Op) -> BooleanDerivativeDFAResult:
        if not isinstance(root, PureOp):
            raise TypeError("a derivative DFA requires pure regex semantics")
        engine = BooleanDerivativeEngine()
        analyses = engine.analyses
        residuals: list[PureOp] = [root]
        states: dict[PureOp, int] = {root: 0}
        rows: list[tuple[Transition, ...]] = []
        outputs: list[bool | None] = []
        pending = deque((root,))

        while pending:
            residual = pending.popleft()
            state = states[residual]
            if state != len(rows):
                raise AssertionError("derivative states must be processed in order")
            outputs.append(True if analyses.get(residual).nullable else None)
            grouped: dict[PureOp, int] = {}
            for symbol in range(BYTE_ALPHABET_SIZE):
                target_residual = engine.derive(residual, symbol)
                if target_residual == NEVER:
                    continue
                grouped[target_residual] = grouped.get(target_residual, 0) | (
                    1 << symbol
                )
            transitions: list[Transition] = []
            for target_residual, bits in grouped.items():
                target = states.get(target_residual)
                if target is None:
                    if (
                        self.max_states is not None
                        and len(residuals) >= self.max_states
                    ):
                        raise BooleanDerivativeStateBudgetExceeded(self.max_states)
                    target = len(residuals)
                    states[target_residual] = target
                    residuals.append(target_residual)
                    pending.append(target_residual)
                transitions.append(
                    Transition(SymbolSet(BYTE_ALPHABET_SIZE, bits), target)
                )
            rows.append(tuple(transitions))

        return BooleanDerivativeDFAResult(
            DFA(BYTE_ALPHABET_SIZE, 0, tuple(outputs), tuple(rows)),
            tuple(residuals),
            engine.cached_derivative_count,
        )


def compile_boolean_dfa(
    root: Op,
    *,
    max_states: int | None = 10_000,
) -> DFA[bool]:
    return BooleanDerivativeDFACompiler(max_states=max_states).compile(root).automaton


__all__ = [
    "BooleanDerivativeDFACompiler",
    "BooleanDerivativeDFAResult",
    "BooleanDerivativeStateBudgetExceeded",
    "compile_boolean_dfa",
]
