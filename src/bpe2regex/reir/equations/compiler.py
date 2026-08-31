"""Compilation from the lazy canonical quotient to regular equations."""

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..automata.canonical_lazy import CanonicalLazyQuotient
from .analysis import LengthGuardednessReport, verify_length_guarded
from .ir import (
    EQUATION_UNIVERSAL,
    EquationDifference,
    EquationTokenSelectorRef,
    RegularEquation,
    RegularEquationIR,
)
from .selectors import (
    DenialConeSelectorMetrics,
    PersistentDenialConeSelectorDAG,
)


@dataclass(frozen=True, slots=True)
class RegularEquationCompilationMetrics:
    state_count: int
    active_token_count: int
    selector: DenialConeSelectorMetrics
    reference_site_count: int
    minimum_guard_bytes: int | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class RegularEquationCompilationResult:
    ir: RegularEquationIR
    guardedness: LengthGuardednessReport
    metrics: RegularEquationCompilationMetrics


class CanonicalRegularEquationCompiler:
    """Compile a lazy quotient without materializing any quotient row."""

    def compile(
        self,
        quotient: CanonicalLazyQuotient,
        tokens: Sequence[bytes | None],
    ) -> RegularEquationCompilationResult:
        started = time.perf_counter()
        vocabulary = tuple(tokens)
        if quotient.alphabet_size != len(vocabulary):
            raise ValueError("lazy quotient and equation vocabulary differ in size")

        base_tokens = vocabulary[: quotient.adjacency.base_token_count]
        if any(token is None or len(token) != 1 for token in base_tokens):
            raise ValueError(
                "regular equations require single-byte base alphabet tokens"
            )
        alphabet = frozenset(token[0] for token in base_tokens if token is not None)
        if len(alphabet) != len(base_tokens):
            raise ValueError("regular-equation base tokens must be unique bytes")

        selectors = PersistentDenialConeSelectorDAG.from_quotient(quotient)
        token_targets: list[int | None] = [None] * len(vocabulary)
        for token in quotient.active_tokens:
            raw_target = quotient.adjacency.token_targets[token]
            if raw_target is None:
                raise AssertionError("an active quotient token must have a target")
            target = quotient.state_map[raw_target]
            if target is None:
                raise AssertionError("an active token target must be reachable")
            token_targets[token] = target

        equations: list[RegularEquation] = []
        for state, selector_id in enumerate(selectors.state_selectors):
            selector = selectors.selectors[selector_id]
            body = (
                EQUATION_UNIVERSAL
                if not selector.cone_ids
                else EquationDifference(
                    EQUATION_UNIVERSAL,
                    EquationTokenSelectorRef(selector_id),
                )
            )
            equations.append(RegularEquation(state, body))

        global_entry = EquationTokenSelectorRef(selectors.global_selector)
        start_selector_id = selectors.state_selectors[quotient.start]
        start_selector = selectors.selectors[start_selector_id]
        entry = (
            global_entry
            if not start_selector.cone_ids
            else EquationDifference(
                global_entry,
                EquationTokenSelectorRef(start_selector_id),
            )
        )
        ir = RegularEquationIR(
            alphabet,
            quotient.start,
            entry,
            tuple(equations),
            vocabulary,
            tuple(token_targets),
            selectors,
        )
        guardedness = verify_length_guarded(ir)
        metrics = RegularEquationCompilationMetrics(
            ir.state_count,
            len(ir.active_tokens),
            selectors.metrics,
            len(guardedness.references),
            guardedness.minimum_guard_bytes,
            time.perf_counter() - started,
        )
        return RegularEquationCompilationResult(ir, guardedness, metrics)


__all__ = [
    "CanonicalRegularEquationCompiler",
    "RegularEquationCompilationMetrics",
    "RegularEquationCompilationResult",
]
