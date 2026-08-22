"""Engine-specific target emitters built on REIR."""

from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal, overload

from .ecmascript import RegexSources as ECMAScriptRegexSources
from .ecmascript import emit_sources as emit_ecmascript_sources
from .pcre2 import PCRE2DAGSource, PCRE2SubroutineDAGEmitter, render_pcre2_dag
from .python import RegexSources as PythonRegexSources
from .python import emit_sources as emit_python_sources


class Compatibility(Enum):
    """Regex syntax and engine constraints targeted by an emitter."""

    PYTHON = "python"
    ECMASCRIPT = "ecmascript"


type RegexSources = PythonRegexSources | ECMAScriptRegexSources


@overload
def emit_regex_sources(
    tokens: Sequence[bytes | None],
    parents: Any,
    compatibility: Literal[Compatibility.PYTHON],
    *,
    base_token_count: int = 256,
) -> PythonRegexSources: ...


@overload
def emit_regex_sources(
    tokens: Sequence[bytes | None],
    parents: Any,
    compatibility: Literal[Compatibility.ECMASCRIPT],
    *,
    base_token_count: int = 256,
) -> ECMAScriptRegexSources: ...


def emit_regex_sources(
    tokens: Sequence[bytes | None],
    parents: Any,
    compatibility: Compatibility,
    *,
    base_token_count: int = 256,
) -> RegexSources:
    """Emit engine-specific regex sources through one compatibility switch."""
    match compatibility:
        case Compatibility.PYTHON:
            return emit_python_sources(
                tokens,
                parents,
                base_token_count=base_token_count,
            )
        case Compatibility.ECMASCRIPT:
            return emit_ecmascript_sources(
                tokens,
                parents,
                base_token_count=base_token_count,
            )
    raise ValueError(f"unsupported regex compatibility: {compatibility!r}")


__all__ = [
    "Compatibility",
    "PCRE2DAGSource",
    "PCRE2SubroutineDAGEmitter",
    "RegexSources",
    "emit_regex_sources",
    "render_pcre2_dag",
]
