from collections.abc import Callable

from .compiler import RegexCompiler
from .lowering import FunctionalOpLowerer, LoweringContext
from .ops import Op
from .source import ByteEscape, RegexSourceLowerer
from .tagged import Tag, TaggedAlternate, TaggedConcat

type TagEmitter = Callable[[int], str]


class TaggedRegexSourceLowerer(RegexSourceLowerer):
    """Extend pure source lowering with ordered transducer-output tags."""

    def __init__(
        self,
        *,
        escape_byte: ByteEscape,
        emit_tag: TagEmitter | None = None,
    ) -> None:
        self.emit_tag = emit_tag
        super().__init__(escape_byte=escape_byte)
        self.register(FunctionalOpLowerer(Tag, self._lower_tag))
        self.register(FunctionalOpLowerer(TaggedConcat, self._lower_concat))
        self.register(FunctionalOpLowerer(TaggedAlternate, self._lower_alternate))

    def _lower_tag(
        self,
        op: Op,
        operands: tuple[str, ...],
        context: LoweringContext,
    ) -> str:
        assert isinstance(op, Tag)
        if self.emit_tag is None:
            raise ValueError("tagged REIR requires an explicit tag emitter")
        return self.emit_tag(op.value)


def render_tagged_regex(
    expression: Op,
    *,
    escape_byte: ByteEscape,
    emit_tag: TagEmitter | None = None,
) -> str:
    """Lower canonical tagged REIR while preserving ordered outputs."""
    compiler = RegexCompiler(
        TaggedRegexSourceLowerer(escape_byte=escape_byte, emit_tag=emit_tag),
    )
    return compiler.compile(expression)


__all__ = [
    "TagEmitter",
    "TaggedRegexSourceLowerer",
    "render_tagged_regex",
]
