from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .analysis import AnalysisManager
from .cost import CostModel
from .ops import Op
from .passes import OperationPass, PassResult, PreservedAnalyses


class CandidateGenerator(ABC):
    """Generate equivalent REIR alternatives without choosing among them."""

    @abstractmethod
    def generate(self, root: Op, analyses: AnalysisManager) -> Iterable[Op]:
        """Yield zero or more alternatives equivalent to ``root``."""


type CandidateCallback = Callable[[Op, AnalysisManager], Iterable[Op]]


class FunctionalCandidateGenerator(CandidateGenerator):
    def __init__(self, callback: CandidateCallback) -> None:
        self.callback = callback

    def generate(self, root: Op, analyses: AnalysisManager) -> Iterable[Op]:
        return self.callback(root, analyses)


@dataclass(frozen=True, slots=True)
class CostedCandidate[CostT]:
    root: Op
    cost: CostT
    ordinal: int


class CandidateSelector[CostT](ABC):
    """Choose one candidate through an explicit cost model."""

    @abstractmethod
    def select(
        self,
        candidates: Iterable[Op],
        cost_model: CostModel[CostT],
        *,
        analyses: AnalysisManager | None = None,
    ) -> CostedCandidate[CostT]:
        """Evaluate candidates and return exactly one winner."""


def _unique_candidates(candidates: Iterable[Op]) -> tuple[Op, ...]:
    unique: list[Op] = []
    hashable_seen: set[Op] = set()
    unhashable_seen: list[Op] = []
    for candidate in candidates:
        if not isinstance(candidate, Op):
            raise TypeError("a candidate generator must yield REIR operations")
        candidate.verify()
        try:
            hash(candidate)
        except TypeError:
            if candidate in unhashable_seen:
                continue
            unhashable_seen.append(candidate)
        else:
            if candidate in hashable_seen:
                continue
            hashable_seen.add(candidate)
        unique.append(candidate)
    return tuple(unique)


class MinimumCostSelector[CostT](CandidateSelector[CostT]):
    """Select the smallest lexicographic key, preserving the first exact tie."""

    def select(
        self,
        candidates: Iterable[Op],
        cost_model: CostModel[CostT],
        *,
        analyses: AnalysisManager | None = None,
    ) -> CostedCandidate[CostT]:
        roots = _unique_candidates(candidates)
        if not roots:
            raise ValueError("candidate selection requires at least one operation")
        manager = AnalysisManager() if analyses is None else analyses
        evaluated = tuple(
            CostedCandidate(
                root,
                cost_model.evaluate(root, analyses=manager),
                ordinal,
            )
            for ordinal, root in enumerate(roots)
        )
        return min(evaluated, key=lambda candidate: cost_model.key(candidate.cost))


class CandidateSelectionPass[CostT](OperationPass):
    """Connect candidate generation, cost evaluation, and stable selection."""

    def __init__(
        self,
        generators: Iterable[CandidateGenerator],
        cost_model: CostModel[CostT],
        selector: CandidateSelector[CostT] | None = None,
    ) -> None:
        self.generators = tuple(generators)
        self.cost_model = cost_model
        self.selector = MinimumCostSelector() if selector is None else selector

    def run(self, root: Op, analyses: AnalysisManager) -> PassResult:
        candidates = [root]
        for generator in self.generators:
            candidates.extend(generator.generate(root, analyses))
        selected = self.selector.select(
            candidates,
            self.cost_model,
            analyses=analyses,
        )
        return PassResult(
            selected.root,
            PreservedAnalyses.all()
            if selected.root is root
            else PreservedAnalyses.none(),
        )


__all__ = [
    "CandidateCallback",
    "CandidateGenerator",
    "CandidateSelectionPass",
    "CandidateSelector",
    "CostedCandidate",
    "FunctionalCandidateGenerator",
    "MinimumCostSelector",
]
