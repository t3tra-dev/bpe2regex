from collections.abc import Callable
from dataclasses import dataclass

type ByteEscape = Callable[[int], str]
type TagEmitter = Callable[[int], str]


@dataclass(frozen=True, slots=True)
class Never:
    """A regular expression that cannot match."""


@dataclass(frozen=True, slots=True)
class Empty:
    """The empty-string regular expression."""


@dataclass(frozen=True, slots=True)
class Literal:
    value: bytes

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("a regex literal must not be empty")


@dataclass(frozen=True, slots=True)
class Tag:
    """A zero-width transducer output to be lowered by a target renderer."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("a regex tag must be non-negative")


@dataclass(frozen=True, slots=True)
class Concat:
    parts: tuple[RegexAST, ...]

    def __post_init__(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("a regex concatenation requires at least two parts")


@dataclass(frozen=True, slots=True)
class Alternate:
    alternatives: tuple[RegexAST, ...]

    def __post_init__(self) -> None:
        if len(self.alternatives) < 2:
            raise ValueError("a regex alternation requires at least two branches")


type RegexAST = Never | Empty | Literal | Tag | Concat | Alternate

NEVER = Never()
EMPTY = Empty()


def literal(value: bytes | bytearray | memoryview) -> RegexAST:
    content = bytes(value)
    return Literal(content) if content else EMPTY


def tagged(value: int) -> Tag:
    return Tag(value)


def concat(*parts: RegexAST) -> RegexAST:
    flattened: list[RegexAST] = []
    pending_literal = bytearray()

    def flush_literal() -> None:
        if pending_literal:
            flattened.append(Literal(bytes(pending_literal)))
            pending_literal.clear()

    for part in parts:
        if isinstance(part, Never):
            return NEVER
        if isinstance(part, Empty):
            continue
        nested = part.parts if isinstance(part, Concat) else (part,)
        for item in nested:
            if isinstance(item, Literal):
                pending_literal.extend(item.value)
            else:
                flush_literal()
                flattened.append(item)
    flush_literal()
    if not flattened:
        return EMPTY
    if len(flattened) == 1:
        return flattened[0]
    return Concat(tuple(flattened))


def alternate(*alternatives: RegexAST) -> RegexAST:
    flattened: list[RegexAST] = []
    for alternative in alternatives:
        if isinstance(alternative, Never):
            continue
        if isinstance(alternative, Alternate):
            flattened.extend(alternative.alternatives)
        else:
            flattened.append(alternative)
    if not flattened:
        return NEVER
    if len(flattened) == 1:
        return flattened[0]
    return Alternate(tuple(flattened))


def render_regex(
    expression: RegexAST,
    *,
    escape_byte: ByteEscape,
    emit_tag: TagEmitter | None = None,
) -> str:
    """Lower an engine-independent AST into a target regex source."""

    match expression:
        case Never():
            return "(?!)"
        case Empty():
            return ""
        case Literal(value):
            return "".join(escape_byte(byte) for byte in value)
        case Tag(value):
            if emit_tag is None:
                raise ValueError("the regex AST contains a tag without a tag emitter")
            return emit_tag(value)
        case Concat(parts):
            return "".join(
                render_regex(part, escape_byte=escape_byte, emit_tag=emit_tag)
                for part in parts
            )
        case Alternate(alternatives):
            return (
                "(?:"
                + "|".join(
                    render_regex(
                        alternative,
                        escape_byte=escape_byte,
                        emit_tag=emit_tag,
                    )
                    for alternative in alternatives
                )
                + ")"
            )


__all__ = [
    "EMPTY",
    "NEVER",
    "Alternate",
    "ByteEscape",
    "Concat",
    "Empty",
    "Literal",
    "Never",
    "RegexAST",
    "Tag",
    "TagEmitter",
    "alternate",
    "concat",
    "literal",
    "render_regex",
    "tagged",
]
