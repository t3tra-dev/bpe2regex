import zlib
from collections.abc import Sequence

from .encoding import Encoding
from .reir.emitter import Compatibility, RegexSources
from .reir.emitter.ecmascript import RegexSources as ECMAScriptRegexSources
from .reir.emitter.python import RegexSources as PythonRegexSources

PYTHON_ARTIFACT_FILENAME = "python.bin"
ECMASCRIPT_ARTIFACT_FILENAME = "ecmascript.bin"

_MAGIC = b"B2RX"
_FORMAT_VERSION = 1
_ENCODING_IDS = {
    Encoding.R50K: 0,
    Encoding.O200K: 1,
    Encoding.P50K: 2,
    Encoding.CL100K: 3,
}
_ENCODINGS = {identifier: encoding for encoding, identifier in _ENCODING_IDS.items()}
_COMPATIBILITY_IDS = {
    Compatibility.PYTHON: 0,
    Compatibility.ECMASCRIPT: 1,
}
_COMPATIBILITIES = {
    identifier: compatibility
    for compatibility, identifier in _COMPATIBILITY_IDS.items()
}


def _encode_uint(value: int) -> bytes:
    if value < 0:
        raise ValueError("binary artifact integers must be non-negative")
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _encode_text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _encode_uint(len(encoded)) + encoded


def _encode_texts(values: Sequence[str]) -> bytes:
    return _encode_uint(len(values)) + b"".join(map(_encode_text, values))


def _rank_byte_width(token_count: int) -> int:
    if token_count <= 0:
        raise ValueError("binary artifact token count must be positive")
    return max(1, ((token_count - 1).bit_length() + 7) // 8)


def _encode_ranks(values: Sequence[int], token_count: int) -> bytes:
    width = _rank_byte_width(token_count)
    encoded = bytearray(_encode_uint(len(values)))
    for rank in values:
        if not 0 <= rank < token_count:
            raise ValueError(f"capture rank is outside the vocabulary: {rank}")
        encoded.extend(rank.to_bytes(width, "little"))
    return bytes(encoded)


def _encode_rank_tables(
    values: Sequence[Sequence[int]],
    token_count: int,
) -> bytes:
    return _encode_uint(len(values)) + b"".join(
        _encode_ranks(table, token_count) for table in values
    )


def _validate_capture_ranks(
    actual: Sequence[int],
    expected: set[int],
    name: str,
) -> None:
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"{name} capture ranks differ from the regex dimensions")


def _compress(payload: bytes) -> bytes:
    compressor = zlib.compressobj(level=9, wbits=-15)
    return compressor.compress(payload) + compressor.flush()


def encode_artifact(
    encoding: Encoding,
    compatibility: Compatibility,
    sources: RegexSources,
    pretokenizer_source: str,
) -> bytes:
    """Encode regex sources as a minimal raw-DEFLATE binary container."""
    expected_reserved_ranks = tuple(
        rank for rank in encoding.reserved_ranks if rank < sources.token_count
    )
    if sources.reserved_ranks != expected_reserved_ranks:
        raise ValueError(
            f"regex reserved ranks differ for {encoding.value}: "
            f"{sources.reserved_ranks!r}"
        )
    header = bytearray(_MAGIC)
    header.append(_FORMAT_VERSION)
    header.append(_ENCODING_IDS[encoding])
    header.append(_COMPATIBILITY_IDS[compatibility])
    header.extend(_encode_uint(sources.token_count))
    header.extend(_encode_uint(sources.base_token_count))
    header.extend(_encode_uint(sources.rank_width))

    match compatibility, sources:
        case Compatibility.PYTHON, PythonRegexSources():
            _validate_capture_ranks(
                sources.byte_capture_ranks,
                set(range(sources.base_token_count)),
                "Python base",
            )
            _validate_capture_ranks(
                sources.merge_capture_ranks,
                set(range(sources.base_token_count, sources.token_count))
                - set(sources.reserved_ranks),
                "Python merge",
            )
            payload = (
                bytes(header)
                + _encode_text(sources.byte_to_rank)
                + _encode_ranks(sources.byte_capture_ranks, sources.token_count)
                + _encode_text(sources.merge_pair)
                + _encode_ranks(sources.merge_capture_ranks, sources.token_count)
                + _encode_text(pretokenizer_source)
            )
        case Compatibility.ECMASCRIPT, ECMAScriptRegexSources():
            frontier_count = len(sources.merge_prefixes)
            if len(sources.merge_patterns) != frontier_count:
                raise ValueError("ECMAScript merge frontier pattern count differs")
            if len(sources.merge_capture_ranks) != frontier_count:
                raise ValueError("ECMAScript merge capture table count differs")
            flattened_ranks = tuple(
                rank
                for capture_ranks in sources.merge_capture_ranks
                for rank in capture_ranks
            )
            _validate_capture_ranks(
                flattened_ranks,
                set(range(sources.base_token_count, sources.token_count))
                - set(sources.reserved_ranks),
                "ECMAScript merge",
            )
            payload = (
                bytes(header)
                + _encode_texts(sources.byte_rank_bits)
                + _encode_texts(sources.merge_prefixes)
                + _encode_texts(sources.merge_patterns)
                + _encode_rank_tables(
                    sources.merge_capture_ranks,
                    sources.token_count,
                )
                + _encode_text(pretokenizer_source)
            )
        case _:
            raise TypeError(
                f"sources {type(sources).__name__} do not match {compatibility.value}"
            )
    return _compress(payload)


class _Reader:
    def __init__(self, value: bytes) -> None:
        try:
            self.value = memoryview(zlib.decompress(value, wbits=-15))
        except zlib.error as error:
            raise ValueError("cannot decompress regex artifact") from error
        self.position = 0

    def read(self, size: int) -> bytes:
        end = self.position + size
        if size < 0 or end > len(self.value):
            raise ValueError("truncated regex artifact")
        result = bytes(self.value[self.position : end])
        self.position = end
        return result

    def uint(self) -> int:
        value = 0
        shift = 0
        while shift <= 63:
            byte = self.read(1)[0]
            value |= (byte & 0x7F) << shift
            if byte < 0x80:
                return value
            shift += 7
        raise ValueError("oversized integer in regex artifact")

    def text(self) -> str:
        try:
            return self.read(self.uint()).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8 in regex artifact") from error

    def texts(self) -> tuple[str, ...]:
        return tuple(self.text() for _ in range(self.uint()))

    def ranks(self, token_count: int) -> tuple[int, ...]:
        count = self.uint()
        width = _rank_byte_width(token_count)
        content = self.read(count * width)
        ranks = tuple(
            int.from_bytes(content[offset : offset + width], "little")
            for offset in range(0, len(content), width)
        )
        if any(rank >= token_count for rank in ranks):
            raise ValueError("capture rank is outside the artifact vocabulary")
        return ranks

    def rank_tables(self, token_count: int) -> tuple[tuple[int, ...], ...]:
        return tuple(self.ranks(token_count) for _ in range(self.uint()))

    def finish(self) -> None:
        if self.position != len(self.value):
            raise ValueError("trailing data in regex artifact")


def _decode_header(
    reader: _Reader,
    expected_compatibility: Compatibility,
) -> tuple[Encoding, int, int, int]:
    if reader.read(len(_MAGIC)) != _MAGIC:
        raise ValueError("invalid regex artifact magic")
    version = reader.read(1)[0]
    if version != _FORMAT_VERSION:
        raise ValueError(f"unsupported regex artifact version: {version}")
    encoding_id = reader.read(1)[0]
    compatibility_id = reader.read(1)[0]
    try:
        encoding = _ENCODINGS[encoding_id]
    except KeyError as error:
        raise ValueError(f"unsupported encoding identifier: {encoding_id}") from error
    try:
        compatibility = _COMPATIBILITIES[compatibility_id]
    except KeyError as error:
        raise ValueError(
            f"unsupported compatibility identifier: {compatibility_id}"
        ) from error
    if compatibility is not expected_compatibility:
        raise ValueError(
            f"expected {expected_compatibility.value}, found {compatibility.value}"
        )
    return encoding, reader.uint(), reader.uint(), reader.uint()


def decode_python_artifact(
    content: bytes,
) -> tuple[Encoding, PythonRegexSources, str]:
    reader = _Reader(content)
    encoding, token_count, base_token_count, rank_width = _decode_header(
        reader,
        Compatibility.PYTHON,
    )
    byte_to_rank = reader.text()
    byte_capture_ranks = reader.ranks(token_count)
    merge_pair = reader.text()
    merge_capture_ranks = reader.ranks(token_count)
    sources = PythonRegexSources(
        byte_to_rank=byte_to_rank,
        merge_pair=merge_pair,
        token_count=token_count,
        base_token_count=base_token_count,
        rank_width=rank_width,
        reserved_ranks=tuple(
            rank for rank in encoding.reserved_ranks if rank < token_count
        ),
        byte_capture_ranks=byte_capture_ranks,
        merge_capture_ranks=merge_capture_ranks,
    )
    pretokenizer_source = reader.text()
    reader.finish()
    return encoding, sources, pretokenizer_source


def decode_ecmascript_artifact(
    content: bytes,
) -> tuple[Encoding, ECMAScriptRegexSources, str]:
    reader = _Reader(content)
    encoding, token_count, base_token_count, rank_width = _decode_header(
        reader,
        Compatibility.ECMASCRIPT,
    )
    byte_rank_bits = reader.texts()
    merge_prefixes = reader.texts()
    merge_patterns = reader.texts()
    merge_capture_ranks = reader.rank_tables(token_count)
    sources = ECMAScriptRegexSources(
        byte_rank_bits=byte_rank_bits,
        merge_prefixes=merge_prefixes,
        merge_patterns=merge_patterns,
        token_count=token_count,
        base_token_count=base_token_count,
        rank_width=rank_width,
        reserved_ranks=tuple(
            rank for rank in encoding.reserved_ranks if rank < token_count
        ),
        merge_capture_ranks=merge_capture_ranks,
    )
    pretokenizer_source = reader.text()
    reader.finish()
    return encoding, sources, pretokenizer_source
