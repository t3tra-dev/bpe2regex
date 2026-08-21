import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from .analysis import AnalysisManager, RegexPropertiesAnalysis
from .lowering import Lowerer
from .ops import Op


@dataclass(frozen=True, order=True, slots=True)
class StructuralCost:
    """Cheap lexicographic cost for pure REIR structure."""

    operation_count: int
    literal_bytes: int


@dataclass(frozen=True, order=True, slots=True)
class LoweredSizeCost:
    """Target size followed by stable structural tie breakers."""

    size: int
    operation_count: int
    literal_bytes: int


type CostCallback[CostT] = Callable[[Op, AnalysisManager], CostT]
type CostKeyValue = tuple[int, ...]
type CostKey[CostT] = Callable[[CostT], CostKeyValue]
type ArtifactSerializer[ResultT] = Callable[[ResultT], bytes]


class CostModel[CostT](ABC):
    """Extensible objective used to evaluate one REIR candidate."""

    @abstractmethod
    def evaluate(
        self,
        root: Op,
        *,
        analyses: AnalysisManager | None = None,
    ) -> CostT:
        """Return a deterministic cost value for ``root``."""

    @abstractmethod
    def key(self, cost: CostT) -> CostKeyValue:
        """Return a totally ordered lexicographic selection key."""


class FunctionalCostModel[CostT](CostModel[CostT]):
    """Adapt an arbitrary target- or artifact-aware callback into a cost model."""

    def __init__(
        self,
        callback: CostCallback[CostT],
        key: CostKey[CostT],
    ) -> None:
        self.callback = callback
        self._key = key

    def evaluate(
        self,
        root: Op,
        *,
        analyses: AnalysisManager | None = None,
    ) -> CostT:
        manager = AnalysisManager() if analyses is None else analyses
        return self.callback(root, manager)

    def key(self, cost: CostT) -> CostKeyValue:
        return self._key(cost)


class StructuralCostModel(CostModel[StructuralCost]):
    """Measure operation occurrences and literal payload in pure REIR."""

    def evaluate(
        self,
        root: Op,
        *,
        analyses: AnalysisManager | None = None,
    ) -> StructuralCost:
        manager = AnalysisManager() if analyses is None else analyses
        properties = manager.get(RegexPropertiesAnalysis, root)
        return StructuralCost(properties.operation_count, properties.literal_bytes)

    def key(self, cost: StructuralCost) -> CostKeyValue:
        return cost.operation_count, cost.literal_bytes


type OutputSize[ResultT] = Callable[[ResultT], int]


class LoweredSizeCostModel[ResultT](CostModel[LoweredSizeCost]):
    """Measure a lowered target while using structural facts as tie breakers."""

    def __init__(
        self,
        lowerer: Lowerer[ResultT],
        measure: OutputSize[ResultT],
    ) -> None:
        self.lowerer = lowerer
        self.measure = measure

    def evaluate(
        self,
        root: Op,
        *,
        analyses: AnalysisManager | None = None,
    ) -> LoweredSizeCost:
        manager = AnalysisManager() if analyses is None else analyses
        properties = manager.get(RegexPropertiesAnalysis, root)
        output = self.lowerer.lower(root, analyses=manager)
        size = self.measure(output)
        if size < 0:
            raise ValueError("a lowered target size must be non-negative")
        return LoweredSizeCost(
            size,
            properties.operation_count,
            properties.literal_bytes,
        )

    def key(self, cost: LoweredSizeCost) -> CostKeyValue:
        return cost.size, cost.operation_count, cost.literal_bytes


def utf8_size(source: str) -> int:
    return len(source.encode("utf-8"))


def raw_deflate_size(source: str | bytes, *, level: int = 9) -> int:
    payload = source.encode("utf-8") if isinstance(source, str) else source
    compressor = zlib.compressobj(level=level, wbits=-15)
    return len(compressor.compress(payload) + compressor.flush())


class SourceSizeCostModel(LoweredSizeCostModel[str]):
    """Minimize UTF-8 target-source bytes."""

    def __init__(self, lowerer: Lowerer[str]) -> None:
        super().__init__(lowerer, utf8_size)


class DeflatedSourceCostModel(LoweredSizeCostModel[str]):
    """Minimize standalone raw-DEFLATE target-source bytes."""

    def __init__(self, lowerer: Lowerer[str], *, level: int = 9) -> None:
        super().__init__(
            lowerer,
            lambda source: raw_deflate_size(source, level=level),
        )


class ArtifactSizeCostModel[ResultT](LoweredSizeCostModel[ResultT]):
    """Minimize a complete serialized artifact produced from lowered output."""

    def __init__(
        self,
        lowerer: Lowerer[ResultT],
        serialize: ArtifactSerializer[ResultT],
    ) -> None:
        super().__init__(lowerer, lambda output: len(serialize(output)))


__all__ = [
    "ArtifactSerializer",
    "ArtifactSizeCostModel",
    "CostCallback",
    "CostKey",
    "CostKeyValue",
    "CostModel",
    "DeflatedSourceCostModel",
    "FunctionalCostModel",
    "LoweredSizeCost",
    "LoweredSizeCostModel",
    "OutputSize",
    "SourceSizeCostModel",
    "StructuralCost",
    "StructuralCostModel",
    "raw_deflate_size",
    "utf8_size",
]
