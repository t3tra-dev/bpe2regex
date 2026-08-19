import hashlib
import json
import unicodedata
from pathlib import Path

RUST_UNICODE_VERSION = "16.0.0"
RUST_REGEX_DEPENDENCIES = {
    "fancy-regex": "0.19.0",
    "regex-automata": "0.4.18",
    "regex-syntax": "0.8.11",
}

# regex-syntax 0.8.11 src/unicode_tables/perl_space.rs, generated from
# Unicode 16.0.0's White_Space property.
WHITE_SPACE_RANGES: tuple[tuple[int, int], ...] = (
    (0x0009, 0x000D),
    (0x0020, 0x0020),
    (0x0085, 0x0085),
    (0x00A0, 0x00A0),
    (0x1680, 0x1680),
    (0x2000, 0x200A),
    (0x2028, 0x2029),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
)


def _ranges_for_categories(
    categories: frozenset[str],
    *,
    prefixes: tuple[str, ...] = (),
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    end = -1
    for codepoint in range(0x110000):
        category = unicodedata.category(chr(codepoint))
        included = category in categories or category.startswith(prefixes)
        if included and start is None:
            start = codepoint
        elif not included and start is not None:
            ranges.append((start, end))
            start = None
        end = codepoint
    if start is not None:
        ranges.append((start, end))
    return tuple(ranges)


def _validate_native_dependency_markers() -> None:
    import tiktoken._tiktoken as native

    if native.__file__ is None:
        raise RuntimeError("cannot locate the tiktoken native module")
    native_path = Path(native.__file__)
    native_data = native_path.read_bytes()
    missing = [
        f"{name}-{dependency_version}"
        for name, dependency_version in RUST_REGEX_DEPENDENCIES.items()
        if f"{name}-{dependency_version}".encode() not in native_data
    ]
    if missing:
        raise RuntimeError(
            "the tiktoken native module does not contain the pinned Rust "
            f"dependency markers: {', '.join(missing)}"
        )


def build_unicode_class_data() -> dict[str, object]:
    """Build explicit ranges matching regex-syntax 0.8.11 / Unicode 16.0.0."""
    if unicodedata.unidata_version != RUST_UNICODE_VERSION:
        raise RuntimeError(
            f"Unicode {RUST_UNICODE_VERSION} is required; "
            f"Python provides {unicodedata.unidata_version}"
        )
    _validate_native_dependency_markers()

    data: dict[str, object] = {
        "unicode_version": RUST_UNICODE_VERSION,
        "rust_regex_dependencies": RUST_REGEX_DEPENDENCIES,
        "letter": _ranges_for_categories(frozenset(), prefixes=("L",)),
        "number": _ranges_for_categories(frozenset(), prefixes=("N",)),
        "o200k_upper": _ranges_for_categories(
            frozenset(("Lu", "Lt", "Lm", "Lo")),
            prefixes=("M",),
        ),
        "o200k_lower": _ranges_for_categories(
            frozenset(("Ll", "Lm", "Lo")),
            prefixes=("M",),
        ),
        "white_space": WHITE_SPACE_RANGES,
    }
    canonical = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
    data["ranges_sha256"] = hashlib.sha256(canonical).hexdigest()
    return data
