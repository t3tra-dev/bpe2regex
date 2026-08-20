from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from .analysis import AnalysisManager, AnalysisType
from .ops import Op
from .rewrite import (
    CANONICALIZATION_PATTERNS,
    OPTIMIZATION_PATTERNS,
    GreedyRewriteDriver,
    RewritePattern,
)


@dataclass(frozen=True, slots=True)
class PreservedAnalyses:
    """Analyses that remain valid after a pass changes its input root."""

    types: frozenset[AnalysisType] = frozenset()
    preserve_all: bool = False

    @classmethod
    def all(cls) -> PreservedAnalyses:
        return cls(preserve_all=True)

    @classmethod
    def none(cls) -> PreservedAnalyses:
        return cls()


@dataclass(frozen=True, slots=True)
class PassResult:
    root: Op
    preserved: PreservedAnalyses = PreservedAnalyses()


class OperationPass(ABC):
    """Base class for one reusable REIR-to-REIR compiler pass."""

    @abstractmethod
    def run(self, root: Op, analyses: AnalysisManager) -> PassResult:
        """Transform a root operation and describe preserved analyses."""


class CanonicalizePass(OperationPass):
    def __init__(
        self,
        patterns: Iterable[RewritePattern] = CANONICALIZATION_PATTERNS,
        *,
        max_rewrites: int = 100_000,
    ) -> None:
        self.driver = GreedyRewriteDriver(patterns, max_rewrites=max_rewrites)

    def run(self, root: Op, analyses: AnalysisManager) -> PassResult:
        rewritten = self.driver.rewrite(root, analyses)
        return PassResult(
            rewritten,
            PreservedAnalyses.all() if rewritten is root else PreservedAnalyses.none(),
        )


class StructureDiscoveryPass(CanonicalizePass):
    """Recover factoring and repetition structure after canonicalization."""

    def __init__(self, *, max_rewrites: int = 100_000) -> None:
        super().__init__(OPTIMIZATION_PATTERNS, max_rewrites=max_rewrites)


class PassManager:
    """Run an ordered transformation pipeline with analysis invalidation."""

    def __init__(self, passes: Iterable[OperationPass] = ()) -> None:
        self._passes = list(passes)

    @property
    def passes(self) -> tuple[OperationPass, ...]:
        return tuple(self._passes)

    def add_pass(self, compiler_pass: OperationPass) -> None:
        self._passes.append(compiler_pass)

    def run(
        self,
        root: Op,
        analyses: AnalysisManager | None = None,
    ) -> Op:
        manager = AnalysisManager() if analyses is None else analyses
        current = root
        for compiler_pass in self._passes:
            result = compiler_pass.run(current, manager)
            if result.root is not current:
                if not result.preserved.preserve_all:
                    manager.invalidate_except(result.preserved.types)
                current = result.root
        return current


__all__ = [
    "CanonicalizePass",
    "OperationPass",
    "PassManager",
    "PassResult",
    "PreservedAnalyses",
    "StructureDiscoveryPass",
]
