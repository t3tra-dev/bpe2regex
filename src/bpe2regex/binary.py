import zlib
from collections.abc import Sequence

from .emitter import Compatibility, RegexSources
from .emitter.ecmascript import RegexSources as ECMAScriptRegexSources
from .emitter.python import RegexSources as PythonRegexSources
from .encoding import Encoding

PYTHON_ARTIFACT_FILENAME = "python.bin"
ECMASCRIPT_ARTIFACT_FILENAME = "ecmascript.bin"

_MAGIC = b"B2RX"
_FORMAT_VERSION = 2
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
            payload = (
                bytes(header)
                + _encode_text(sources.byte_to_rank)
                + _encode_text(sources.merge_pair)
                + _encode_text(pretokenizer_source)
            )
        case Compatibility.ECMASCRIPT, ECMAScriptRegexSources():
            if sources.merge_bucket_count != len(sources.merge_buckets):
                raise ValueError("ECMAScript merge bucket count differs")
            payload = (
                bytes(header)
                + _encode_texts(sources.byte_rank_bits)
                + _encode_texts(sources.merge_buckets)
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
    sources = PythonRegexSources(
        byte_to_rank=reader.text(),
        merge_pair=reader.text(),
        token_count=token_count,
        base_token_count=base_token_count,
        rank_width=rank_width,
        reserved_ranks=tuple(
            rank for rank in encoding.reserved_ranks if rank < token_count
        ),
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
    merge_buckets = reader.texts()
    sources = ECMAScriptRegexSources(
        byte_rank_bits=byte_rank_bits,
        merge_buckets=merge_buckets,
        merge_bucket_count=len(merge_buckets),
        token_count=token_count,
        base_token_count=base_token_count,
        rank_width=rank_width,
        reserved_ranks=tuple(
            rank for rank in encoding.reserved_ranks if rank < token_count
        ),
    )
    pretokenizer_source = reader.text()
    reader.finish()
    return encoding, sources, pretokenizer_source
