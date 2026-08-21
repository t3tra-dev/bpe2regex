from collections.abc import Iterable

from .analysis import AnalysisManager
from .builder import DEFAULT_BUILDER, RegexBuilder
from .derivative import DerivativeEngine
from .ops import Op
from .search import CandidateGenerator


class DerivativeFactoringGenerator(CandidateGenerator):
    """Offer one grouped Brzozowski expansion without forcing a rewrite."""

    def __init__(self, *, builder: RegexBuilder = DEFAULT_BUILDER) -> None:
        self.builder = builder

    def generate(self, root: Op, analyses: AnalysisManager) -> Iterable[Op]:
        candidate = DerivativeEngine(
            builder=self.builder,
            analyses=analyses,
        ).expand(root)
        return () if candidate == root else (candidate,)


__all__ = ["DerivativeFactoringGenerator"]
