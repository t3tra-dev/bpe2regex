"""Source lowering for byte regexes with one observable boundary marker."""

from collections.abc import Callable

from .analysis import AnalysisManager
from .compiler import RegexCompiler
from .lowering import FunctionalOpLowerer, LoweringContext
from .marked import Boundary, verify_single_boundary
from .ops import Op
from .source import ByteEscape, RegexSourceLowerer

type BoundaryEmitter = Callable[[], str]


class MarkedRegexSourceLowerer(RegexSourceLowerer):
    """Erase the augmented-alphabet marker to target-specific source."""

    def __init__(
        self,
        *,
        escape_byte: ByteEscape,
        emit_boundary: BoundaryEmitter,
    ) -> None:
        self.emit_boundary = emit_boundary
        super().__init__(escape_byte=escape_byte)
        self.register(FunctionalOpLowerer(Boundary, self._lower_boundary))

    def _lower_boundary(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        assert isinstance(op, Boundary)
        return self.emit_boundary()

    def lower(
        self,
        root: Op,
        *,
        analyses: AnalysisManager | None = None,
    ) -> str:
        manager = AnalysisManager() if analyses is None else analyses
        verify_single_boundary(root, analyses=manager)
        return super().lower(root, analyses=manager)


def render_marked_regex(
    expression: Op,
    *,
    escape_byte: ByteEscape,
    emit_boundary: BoundaryEmitter,
) -> str:
    """Validate and lower a single-boundary marked language."""
    return RegexCompiler(
        MarkedRegexSourceLowerer(
            escape_byte=escape_byte,
            emit_boundary=emit_boundary,
        )
    ).compile(expression)


__all__ = [
    "BoundaryEmitter",
    "MarkedRegexSourceLowerer",
    "render_marked_regex",
]
