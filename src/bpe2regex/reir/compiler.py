from collections.abc import Iterable
from dataclasses import dataclass

from .analysis import AnalysisManager
from .lowering import Lowerer
from .ops import Op
from .passes import OperationPass, PassManager


@dataclass(frozen=True, slots=True)
class CompilationResult[ResultT]:
    ir: Op
    output: ResultT


class RegexCompiler[ResultT]:
    """Compose REIR transformation passes with one final target lowerer."""

    def __init__(
        self,
        lowerer: Lowerer[ResultT],
        passes: Iterable[OperationPass] = (),
    ) -> None:
        self.lowerer = lowerer
        self.pass_manager = PassManager(passes)

    def run(self, root: Op) -> CompilationResult[ResultT]:
        analyses = AnalysisManager()
        transformed = self.pass_manager.run(root, analyses)
        output = self.lowerer.lower(transformed, analyses=analyses)
        return CompilationResult(transformed, output)

    def compile(self, root: Op) -> ResultT:
        return self.run(root).output


__all__ = ["CompilationResult", "RegexCompiler"]
