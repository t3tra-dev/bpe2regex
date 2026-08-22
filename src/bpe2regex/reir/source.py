from collections.abc import Callable

from .compiler import RegexCompiler
from .cost import DeflatedSourceCostModel
from .derivative_search import DerivativeFactoringGenerator
from .lowering import FunctionalOpLowerer, LoweringContext, RuleBasedLowerer
from .ops import Alternate, CharSet, Concat, Epsilon, Literal, Never, Op, Repeat
from .passes import StructureDiscoveryPass
from .search import CandidateSelectionPass

type ByteEscape = Callable[[int], str]


class RegexSourceLowerer(RuleBasedLowerer[str]):
    """Lower the seven pure REIR operations to regex source."""

    def __init__(self, *, escape_byte: ByteEscape) -> None:
        self.escape_byte = escape_byte
        super().__init__(
            (
                FunctionalOpLowerer(Never, self._lower_never),
                FunctionalOpLowerer(Epsilon, self._lower_epsilon),
                FunctionalOpLowerer(CharSet, self._lower_charset),
                FunctionalOpLowerer(Literal, self._lower_literal),
                FunctionalOpLowerer(Concat, self._lower_concat),
                FunctionalOpLowerer(Alternate, self._lower_alternate),
                FunctionalOpLowerer(Repeat, self._lower_repeat),
            )
        )

    def _lower_never(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        return "(?!)"

    def _lower_epsilon(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        return ""

    def _lower_charset(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        assert isinstance(op, CharSet)
        if op.bits.bit_count() == 1:
            return self.escape_byte(next(iter(op.symbols)))
        fragments: list[str] = []
        for start, end in op.intervals:
            start_source = self.escape_byte(start)
            if start == end:
                fragments.append(start_source)
                continue
            end_source = self.escape_byte(end)
            range_source = f"{start_source}-{end_source}"
            expanded_source = "".join(
                self.escape_byte(symbol) for symbol in range(start, end + 1)
            )
            fragments.append(
                range_source
                if len(range_source) < len(expanded_source)
                else expanded_source
            )
        return "[" + "".join(fragments) + "]"

    def _lower_literal(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        assert isinstance(op, Literal)
        return "".join(self.escape_byte(byte) for byte in op.value)

    def _lower_concat(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        return "".join(operands)

    def _lower_alternate(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        return "(?:" + "|".join(operands) + ")"

    def _lower_repeat(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        assert isinstance(op, Repeat)
        match op.min, op.max:
            case 0, 1:
                quantifier = "?"
            case 0, None:
                quantifier = "*"
            case 1, None:
                quantifier = "+"
            case minimum, maximum if minimum == maximum:
                quantifier = f"{{{minimum}}}"
            case minimum, None:
                quantifier = f"{{{minimum},}}"
            case minimum, maximum:
                quantifier = f"{{{minimum},{maximum}}}"
        body_source = operands[0]
        if isinstance(op.body, CharSet) or (
            isinstance(op.body, Literal) and len(op.body.value) == 1
        ):
            return body_source + quantifier
        if isinstance(op.body, Alternate):
            return body_source + quantifier
        return f"(?:{body_source}){quantifier}"


def render_regex(expression: Op, *, escape_byte: ByteEscape) -> str:
    """Optimize pure REIR and compile it into target regex source."""
    lowerer = RegexSourceLowerer(escape_byte=escape_byte)
    compiler = RegexCompiler(
        lowerer,
        passes=(
            StructureDiscoveryPass(),
            CandidateSelectionPass(
                (DerivativeFactoringGenerator(),),
                DeflatedSourceCostModel(lowerer),
            ),
        ),
    )
    return compiler.compile(expression)


__all__ = [
    "ByteEscape",
    "RegexSourceLowerer",
    "render_regex",
]
