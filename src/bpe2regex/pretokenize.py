import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .encoding import Encoding
from .unicode_data import RUST_REGEX_DEPENDENCIES

Range = tuple[int, int]


def _validated_ranges(value: object, name: str) -> tuple[Range, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} ranges must be an array")
    ranges: list[Range] = []
    previous_end = -1
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(isinstance(endpoint, int) for endpoint in item)
        ):
            raise TypeError(f"invalid {name} range: {item!r}")
        start, end = item
        if not (0 <= start <= end <= 0x10FFFF):
            raise ValueError(f"invalid {name} range endpoints: {item!r}")
        if start <= previous_end:
            raise ValueError(f"overlapping or unsorted {name} ranges")
        ranges.append((start, end))
        previous_end = end
    return tuple(ranges)


def _class_body(ranges: Sequence[Range]) -> str:
    parts: list[str] = []
    for start, end in ranges:
        literal_start = chr(start)
        if start == end:
            parts.append(literal_start)
        else:
            parts.append(f"{literal_start}-{chr(end)}")
    return "".join(parts)


def _canonical_range_data(data: Mapping[str, Any]) -> bytes:
    content = {key: value for key, value in data.items() if key != "ranges_sha256"}
    return json.dumps(content, separators=(",", ":"), sort_keys=True).encode()


@dataclass(frozen=True, slots=True)
class PreToken:
    text: str
    span: tuple[int, int]


class PreTokenizer[EncodingT: Encoding]:
    def __init__(self, encoding: EncodingT, data: Mapping[str, Any]) -> None:
        self.encoding = encoding
        if data.get("unicode_version") != "16.0.0":
            raise ValueError(
                f"unexpected Unicode version: {data.get('unicode_version')!r}"
            )
        if data.get("rust_regex_dependencies") != RUST_REGEX_DEPENDENCIES:
            raise ValueError("unexpected Rust regex dependency versions")
        expected_digest = data.get("ranges_sha256")
        actual_digest = hashlib.sha256(_canonical_range_data(data)).hexdigest()
        if expected_digest != actual_digest:
            raise ValueError("Unicode range data SHA-256 mismatch")

        self.letter_ranges = _validated_ranges(data.get("letter"), "letter")
        self.number_ranges = _validated_ranges(data.get("number"), "number")
        self.white_space_ranges = _validated_ranges(
            data.get("white_space"), "white_space"
        )
        self.o200k_upper_ranges = _validated_ranges(
            data.get("o200k_upper"), "o200k_upper"
        )
        self.o200k_lower_ranges = _validated_ranges(
            data.get("o200k_lower"), "o200k_lower"
        )
        self.metadata = dict(data)

        letter_body = _class_body(self.letter_ranges)
        number_body = _class_body(self.number_ranges)
        white_space_body = _class_body(self.white_space_ranges)
        letter = f"[{letter_body}]"
        number = f"[{number_body}]"
        white_space = f"[{white_space_body}]"
        not_white_space = f"[^{white_space_body}]"
        not_letter_number_space = f"[^{letter_body}{number_body}{white_space_body}]"

        match encoding:
            case Encoding.R50K | Encoding.P50K:
                self.source = (
                    r"'(?:[sdmt]|ll|ve|re)"
                    rf"| ?{letter}+"
                    rf"| ?{number}+"
                    rf"| ?{not_letter_number_space}+"
                    rf"|{white_space}+(?![\s\S])"
                    rf"|{white_space}+(?!{not_white_space})"
                    rf"|{white_space}"
                )
            case Encoding.CL100K:
                prefix = f"[^\r\n{letter_body}{number_body}]"
                contraction = r"'(?:[sSſdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])"
                self.source = (
                    rf"{contraction}"
                    rf"|{prefix}?{letter}+"
                    rf"|{number}{{1,3}}"
                    rf"| ?{not_letter_number_space}+[\r\n]*"
                    rf"|{white_space}+(?![\s\S])"
                    rf"|{white_space}*[\r\n]"
                    rf"|{white_space}+(?!{not_white_space})"
                    rf"|{white_space}"
                )
            case Encoding.O200K:
                upper = f"[{_class_body(self.o200k_upper_ranges)}]"
                lower = f"[{_class_body(self.o200k_lower_ranges)}]"
                prefix = f"[^\r\n{letter_body}{number_body}]"
                contraction = (
                    r"(?:'[sSſ]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|"
                    r"'[lL][lL]|'[dD])?"
                )
                self.source = (
                    rf"{prefix}?{upper}*{lower}+{contraction}"
                    rf"|{prefix}?{upper}+{lower}*{contraction}"
                    rf"|{number}{{1,3}}"
                    rf"| ?{not_letter_number_space}+[\r\n/]*"
                    rf"|{white_space}*[\r\n]+"
                    rf"|{white_space}+(?!{not_white_space})"
                    rf"|{white_space}+"
                )
        self.pattern = re.compile(self.source)

    @classmethod
    def from_source(
        cls,
        encoding: EncodingT,
        source: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> PreTokenizer[EncodingT]:
        """Compile an already expanded stdlib-re pre-tokenizer source."""
        if not isinstance(source, str):
            raise TypeError("pre-tokenizer source must be str")
        instance = cls.__new__(cls)
        instance.encoding = encoding
        instance.letter_ranges = ()
        instance.number_ranges = ()
        instance.white_space_ranges = ()
        instance.o200k_upper_ranges = ()
        instance.o200k_lower_ranges = ()
        instance.metadata = dict(metadata or {})
        instance.source = source
        instance.pattern = re.compile(source)
        if instance.pattern.match("") is not None:
            raise ValueError("pre-tokenizer pattern must not match an empty string")
        return instance

    @classmethod
    def from_file(
        cls,
        encoding: EncodingT,
        path: str | Path,
    ) -> PreTokenizer[EncodingT]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read Unicode class artifact: {path}") from error
        if not isinstance(data, dict):
            raise TypeError("Unicode class artifact must be a JSON object")
        return cls(encoding, data)

    def finditer(self, text: str) -> Iterable[PreToken]:
        if not isinstance(text, str):
            raise TypeError("pre-tokenizer input must be str")
        position = 0
        for match in self.pattern.finditer(text):
            if match.start() != position or match.end() <= match.start():
                raise RuntimeError(
                    f"{self.encoding.value} pre-tokenizer left a gap "
                    f"at character {position}"
                )
            yield PreToken(match.group(), match.span())
            position = match.end()
        if position != len(text):
            raise RuntimeError(
                f"{self.encoding.value} pre-tokenizer stopped at character {position}"
            )

    def split(self, text: str) -> list[str]:
        return [piece.text for piece in self.finditer(text)]
