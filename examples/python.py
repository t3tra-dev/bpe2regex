import argparse
import heapq
import logging
import re
import sys
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"B2RX"
FORMAT_VERSION = 2
ENCODINGS = {
    0: "r50k_base",
    1: "o200k_base",
    2: "p50k_base",
    3: "cl100k_base",
}
RESERVED_RANKS = {"p50k_base": (50_256,)}
PYTHON_COMPATIBILITY = 0
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / ".artifacts/r50k/python.bin"
LOGGER = logging.getLogger("bpe2regex.example.python")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class BinaryReader:
    def __init__(self, path: Path) -> None:
        try:
            self.value = memoryview(zlib.decompress(path.read_bytes(), wbits=-15))
        except (OSError, zlib.error) as error:
            raise ValueError(f"cannot read compressed artifact: {path}") from error
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

    def finish(self) -> None:
        if self.position != len(self.value):
            raise ValueError("trailing data in regex artifact")


def decode_artifact(path: Path) -> tuple[str, str, str, str, int, int, int]:
    reader = BinaryReader(path)
    if reader.read(len(MAGIC)) != MAGIC:
        raise ValueError("invalid regex artifact magic")
    version = reader.read(1)[0]
    encoding = reader.read(1)[0]
    compatibility = reader.read(1)[0]
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported regex artifact version: {version}")
    if encoding not in ENCODINGS:
        raise ValueError(f"unsupported encoding identifier: {encoding}")
    if compatibility != PYTHON_COMPATIBILITY:
        raise ValueError(f"unexpected compatibility identifier: {compatibility}")
    token_count = reader.uint()
    base_token_count = reader.uint()
    rank_width = reader.uint()
    byte_to_rank = reader.text()
    merge_pair = reader.text()
    pretokenizer = reader.text()
    reader.finish()
    if (
        token_count <= base_token_count
        or base_token_count != 256
        or rank_width != len(str(token_count - 1))
    ):
        raise ValueError("invalid regex artifact dimensions")
    return (
        ENCODINGS[encoding],
        byte_to_rank,
        merge_pair,
        pretokenizer,
        token_count,
        base_token_count,
        rank_width,
    )


@dataclass(frozen=True, slots=True)
class TokenMatch:
    source: bytes
    token_ids: list[int]
    spans: list[tuple[int, int]]

    @property
    def captures(self) -> list[bytes]:
        return [self.source[start:end] for start, end in self.spans]


def group_ranks(
    pattern: re.Pattern[bytes],
    prefix: str,
    expected: Iterable[int],
) -> tuple[int, ...]:
    by_index = [-1] * (pattern.groups + 1)
    observed: set[int] = set()
    for name, index in pattern.groupindex.items():
        if not name.startswith(prefix) or not name[len(prefix) :].isdecimal():
            raise ValueError(f"invalid rank capture: {name}")
        rank = int(name[len(prefix) :])
        by_index[index] = rank
        observed.add(rank)
    if pattern.groups != len(pattern.groupindex) or observed != set(expected):
        raise ValueError("rank capture groups differ from artifact dimensions")
    return tuple(by_index)


class RegexProgram:
    def __init__(
        self,
        byte_source: str,
        merge_source: str,
        token_count: int,
        base_token_count: int,
        rank_width: int,
        reserved_ranks: tuple[int, ...] = (),
    ) -> None:
        self.byte_pattern = re.compile(byte_source.encode("ascii"))
        self.merge_pattern = re.compile(merge_source.encode("ascii"))
        self.base_ranks = group_ranks(
            self.byte_pattern,
            "b",
            range(base_token_count),
        )
        self.merge_ranks = group_ranks(
            self.merge_pattern,
            "m",
            (
                rank
                for rank in range(base_token_count, token_count)
                if rank not in reserved_ranks
            ),
        )
        self.token_count = token_count
        self.rank_width = rank_width
        self.merge_cache: dict[tuple[int, int], int] = {}

    def merge_rank(self, left: int, right: int) -> int | None:
        key = (left, right)
        cached = self.merge_cache.get(key)
        if cached is not None:
            return cached
        encoded = f"{left:0{self.rank_width}d},{right:0{self.rank_width}d}".encode()
        match = self.merge_pattern.fullmatch(encoded)
        if match is None:
            return None
        if match.lastindex is None:
            raise ValueError("merge regex matched without a rank capture")
        rank = self.merge_ranks[match.lastindex]
        if rank < 0:
            raise ValueError("merge regex selected an unknown rank")
        self.merge_cache[key] = rank
        return rank

    def fullmatch(self, source: bytes) -> TokenMatch:
        token_ids: list[int] = []
        position = 0
        for match in self.byte_pattern.finditer(source):
            if match.start() != position or match.end() != position + 1:
                raise ValueError("byte regex left a gap")
            if match.lastindex is None:
                raise ValueError("byte regex matched without a rank capture")
            token_ids.append(self.base_ranks[match.lastindex])
            position += 1
        if position != len(source):
            raise ValueError("byte regex did not cover the input")
        if not token_ids:
            return TokenMatch(source, [], [])

        count = len(token_ids)
        starts = list(range(count))
        ends = list(range(1, count + 1))
        previous = [-1, *range(count - 1)]
        following = [*range(1, count), -1]
        versions = [0] * count
        candidates: list[tuple[int, int, int, int, int]] = []

        def push(left_index: int) -> None:
            right_index = following[left_index]
            if right_index < 0:
                return
            rank = self.merge_rank(token_ids[left_index], token_ids[right_index])
            if rank is not None:
                heapq.heappush(
                    candidates,
                    (
                        rank,
                        left_index,
                        versions[left_index],
                        right_index,
                        versions[right_index],
                    ),
                )

        for index in range(count - 1):
            push(index)
        while candidates:
            rank, left, left_version, right, right_version = heapq.heappop(candidates)
            if (
                versions[left] != left_version
                or versions[right] != right_version
                or following[left] != right
                or previous[right] != left
            ):
                continue
            token_ids[left] = rank
            ends[left] = ends[right]
            next_index = following[right]
            following[left] = next_index
            if next_index >= 0:
                previous[next_index] = left
            following[right] = -2
            previous[right] = -2
            versions[left] += 1
            versions[right] += 1
            if previous[left] >= 0:
                push(previous[left])
            push(left)

        final_ids: list[int] = []
        spans: list[tuple[int, int]] = []
        index = 0
        while index >= 0:
            final_ids.append(token_ids[index])
            spans.append((starts[index], ends[index]))
            index = following[index]
        return TokenMatch(source, final_ids, spans)


def repair_surrogates(text: str) -> str:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return text


class Tokenizer:
    def __init__(self, artifact: Path) -> None:
        (
            self.encoding,
            byte_source,
            merge_source,
            pretokenizer,
            token_count,
            base_count,
            width,
        ) = decode_artifact(artifact)
        self.program = RegexProgram(
            byte_source,
            merge_source,
            token_count,
            base_count,
            width,
            RESERVED_RANKS.get(self.encoding, ()),
        )
        self.pretokenizer = re.compile(pretokenizer)

    def encode_ordinary(self, text: str) -> list[int]:
        source = repair_surrogates(text)
        token_ids: list[int] = []
        position = 0
        for match in self.pretokenizer.finditer(source):
            if match.start() != position or match.end() <= match.start():
                raise ValueError(f"pre-tokenizer left a gap at {position}")
            token_ids.extend(self.program.fullmatch(match.group().encode()).token_ids)
            position = match.end()
        if position != len(source):
            raise ValueError(f"pre-tokenizer stopped at {position}")
        return token_ids

    def tokenize_ordinary(self, text: str) -> list[bytes]:
        source = repair_surrogates(text)
        tokens: list[bytes] = []
        position = 0
        for match in self.pretokenizer.finditer(source):
            if match.start() != position or match.end() <= match.start():
                raise ValueError(f"pre-tokenizer left a gap at {position}")
            token_match = self.program.fullmatch(match.group().encode())
            tokens.extend(token_match.captures)
            position = match.end()
        if position != len(source):
            raise ValueError(f"pre-tokenizer stopped at {position}")
        return tokens


def display_token(token: bytes) -> str:
    try:
        text = token.decode("utf-8")
    except UnicodeDecodeError:
        escaped_bytes = "".join(f"\\x{value:02x}" for value in token)
        return f'"{escaped_bytes}"'

    escaped: list[str] = []
    for character in text:
        codepoint = ord(character)
        match character:
            case "\\":
                escaped.append("\\\\")
            case '"':
                escaped.append('\\"')
            case "\n":
                escaped.append("\\n")
            case "\r":
                escaped.append("\\r")
            case "\t":
                escaped.append("\\t")
            case _ if codepoint < 0x20 or 0x7F <= codepoint <= 0x9F:
                escaped.append(f"\\u{codepoint:04x}")
            case _ if codepoint in (0x2028, 0x2029):
                escaped.append(f"\\u{codepoint:04x}")
            case _:
                escaped.append(character)
    return f'"{"".join(escaped)}"'


def format_tokens(tokens: list[bytes]) -> str:
    return "[" + ", ".join(map(display_token, tokens)) + "]"


R50K_CASES = (
    ("", []),
    ("hello world", [31373, 995]),
    (
        " 日本語とEnglish 12345!\n",
        [10545, 245, 98, 17312, 105, 45739, 252, 30201, 15823, 17031, 2231, 0, 198],
    ),
    ("\x1ca", [216, 64]),
    ("a\n\n", [64, 628]),
    ("<|endoftext|>", [27, 91, 437, 1659, 5239, 91, 29]),
    ("a\ud800b", [64, 4210, 65]),
)

P50K_CASES = (
    ("", []),
    ("hello world", [31373, 995]),
    (
        " 日本語とEnglish 12345!\n",
        [10545, 245, 98, 17312, 105, 45739, 252, 30201, 15823, 17031, 2231, 0, 198],
    ),
    ("\x1ca", [216, 64]),
    ("a\n\n", [64, 628]),
    ("<|endoftext|>", [27, 91, 437, 1659, 5239, 91, 29]),
    ("a\ud800b", [64, 4210, 65]),
    ("hello    world", [31373, 50258, 995]),
)


CL100K_CASES = (
    ("", []),
    ("hello world", [15339, 1917]),
    (
        " 日本語とEnglish 12345!\n",
        [76502, 22656, 45918, 252, 19732, 23392, 220, 4513, 1774, 4999],
    ),
    ("\x1ca", [216, 64]),
    ("a\n\n", [64, 271]),
    ("<|endoftext|>", [27, 91, 8862, 728, 428, 91, 29]),
    ("a\ud800b", [64, 5809, 65]),
    ("!hello1234567", [0, 15339, 4513, 10961, 22]),
)


O200K_CASES = (
    ("", []),
    ("hello world", [24912, 2375]),
    (
        " 日本語とEnglish 12345!\n",
        [17428, 40909, 5330, 28881, 220, 7633, 2548, 4175],
    ),
    ("\x1ca", [216, 64]),
    ("a\n\n", [64, 279]),
    ("<|endoftext|>", [27, 91, 419, 1440, 919, 91, 29]),
    ("a\ud800b", [64, 3251, 65]),
)

CASES_BY_ENCODING = {
    "r50k_base": R50K_CASES,
    "p50k_base": P50K_CASES,
    "cl100k_base": CL100K_CASES,
    "o200k_base": O200K_CASES,
}


def verify(tokenizer: Tokenizer) -> None:
    cases = CASES_BY_ENCODING[tokenizer.encoding]
    for text, expected in cases:
        actual = tokenizer.encode_ordinary(text)
        if actual != expected:
            raise AssertionError(
                f"token IDs differ for {text!r}: {actual} != {expected}"
            )
        reconstructed = b"".join(tokenizer.tokenize_ordinary(text))
        if reconstructed != repair_surrogates(text).encode():
            raise AssertionError(f"token bytes differ for {text!r}")
    source = b"hello world"
    match = tokenizer.program.fullmatch(source)
    expected_hello = dict(cases)["hello world"]
    if match.token_ids != expected_hello or b"".join(match.captures) != source:
        raise AssertionError("byte captures do not reconstruct hello world")
    LOGGER.info("python %s: ok (%d text cases)", tokenizer.encoding, len(cases))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="*")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(message)s",
        stream=sys.stdout,
    )
    if arguments.quiet:
        logging.disable(logging.CRITICAL)
    LOGGER.debug("loading artifact: %s", arguments.artifact)
    tokenizer = Tokenizer(arguments.artifact)
    if arguments.verify:
        verify(tokenizer)
    else:
        text = " ".join(arguments.text) if arguments.text else "hello world"
        sys.stdout.write(format_tokens(tokenizer.tokenize_ordinary(text)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
