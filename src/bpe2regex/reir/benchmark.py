import time
from collections.abc import Callable
from dataclasses import dataclass

from .analysis import AnalysisManager, RegexPropertiesAnalysis
from .compiler import CompilationResult, RegexCompiler
from .cost import raw_deflate_size, utf8_size
from .ops import Op

type Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class CompilationMetrics:
    """One reproducible structural/size sample plus wall-clock timing."""

    iterations: int
    seconds_per_iteration: float
    operation_count: int
    literal_bytes: int
    source_characters: int
    source_bytes: int
    deflated_source_bytes: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    compilation: CompilationResult[str]
    metrics: CompilationMetrics


def benchmark_compiler(
    root: Op,
    compiler: RegexCompiler[str],
    *,
    iterations: int = 1,
    clock: Clock = time.perf_counter,
) -> BenchmarkResult:
    """Compile repeatedly and measure the final pure REIR and regex source."""
    if iterations <= 0:
        raise ValueError("benchmark iterations must be positive")

    started = clock()
    first = compiler.run(root)
    for _ in range(iterations - 1):
        compilation = compiler.run(root)
        if compilation != first:
            raise ValueError("regex compilation is not deterministic")
    elapsed = clock() - started
    if elapsed < 0:
        raise ValueError("benchmark clock moved backwards")

    properties = AnalysisManager().get(RegexPropertiesAnalysis, first.ir)
    source = first.output
    metrics = CompilationMetrics(
        iterations=iterations,
        seconds_per_iteration=elapsed / iterations,
        operation_count=properties.operation_count,
        literal_bytes=properties.literal_bytes,
        source_characters=len(source),
        source_bytes=utf8_size(source),
        deflated_source_bytes=raw_deflate_size(source),
    )
    return BenchmarkResult(first, metrics)


__all__ = [
    "BenchmarkResult",
    "Clock",
    "CompilationMetrics",
    "benchmark_compiler",
]
