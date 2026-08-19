import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .encoding import Encoding

TIKTOKEN_VERSION = "0.14.0"
BASE_TOKEN_COUNT = 256


@dataclass(frozen=True, slots=True)
class VocabularySpec:
    source_sha256: str


_SPECS = MappingProxyType(
    {
        Encoding.R50K: VocabularySpec(
            source_sha256=(
                "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"
            ),
        ),
        Encoding.P50K: VocabularySpec(
            source_sha256=(
                "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"
            ),
        ),
        Encoding.CL100K: VocabularySpec(
            source_sha256=(
                "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
            ),
        ),
        Encoding.O200K: VocabularySpec(
            source_sha256=(
                "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
            ),
        ),
    }
)


def vocabulary_spec(encoding: Encoding) -> VocabularySpec:
    try:
        return _SPECS[encoding]
    except KeyError as error:
        raise ValueError(f"unsupported vocabulary encoding: {encoding!r}") from error


@dataclass(frozen=True, slots=True)
class Vocabulary[EncodingT: Encoding]:
    encoding: EncodingT
    tokens: tuple[bytes | None, ...]
    rank_of: Mapping[bytes, int]
    source_sha256: str
    native_module_sha256: str
    tiktoken_version: str

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def mergeable_token_count(self) -> int:
        return len(self.rank_of)


def vocabulary_sha256(tokens: Sequence[bytes | None]) -> str:
    """Hash a rank-ordered vocabulary without depending on a container format."""
    digest = hashlib.sha256()
    for rank, token in enumerate(tokens):
        digest.update(rank.to_bytes(4, "little"))
        if token is None:
            digest.update((0xFFFF).to_bytes(2, "little"))
            continue
        digest.update(len(token).to_bytes(2, "little"))
        digest.update(token)
    return digest.hexdigest()


def load_vocabulary[EncodingT: Encoding](
    encoding: EncodingT,
) -> Vocabulary[EncodingT]:
    """Load and validate the pinned vocabulary for an encoding variant."""
    spec = vocabulary_spec(encoding)
    installed_version = version("tiktoken")
    if installed_version != TIKTOKEN_VERSION:
        raise RuntimeError(
            "artifact generation requires "
            f"tiktoken=={TIKTOKEN_VERSION}; found {installed_version}"
        )

    # Kept inside the build-only function so importing bpe2regex does not import
    # tiktoken (and therefore its third-party regex dependency).
    import tiktoken
    import tiktoken._tiktoken as native

    tiktoken_encoding = tiktoken.get_encoding(encoding.value)
    ranks = dict(tiktoken_encoding._mergeable_ranks)

    if len(ranks) != encoding.mergeable_token_count:
        raise ValueError(f"unexpected {encoding.value} vocabulary size: {len(ranks)}")
    expected_ranks = set(range(encoding.token_count)) - set(encoding.reserved_ranks)
    if set(ranks.values()) != expected_ranks:
        raise ValueError(
            f"{encoding.value} ranks differ from its declared mergeable rank space"
        )

    tokens_by_rank: list[bytes | None] = [None] * encoding.token_count
    for token, rank in ranks.items():
        tokens_by_rank[rank] = token
    tokens = tuple(tokens_by_rank)

    base_tokens = tokens[: encoding.base_token_count]
    expected_base_tokens = {bytes((value,)) for value in range(256)}
    if any(token is None or len(token) != 1 for token in base_tokens):
        raise ValueError(
            f"the first 256 {encoding.value} tokens are not all one byte long"
        )
    if set(base_tokens) != expected_base_tokens:
        raise ValueError(
            f"the first 256 {encoding.value} tokens do not cover every byte"
        )
    if native.__file__ is None:
        raise RuntimeError("cannot locate the tiktoken native module")

    return Vocabulary[EncodingT](
        encoding=encoding,
        tokens=tokens,
        rank_of=MappingProxyType(ranks),
        source_sha256=spec.source_sha256,
        native_module_sha256=hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest(),
        tiktoken_version=installed_version,
    )


def reference_bpe_ids(
    piece: bytes,
    tokens: Sequence[bytes | None],
    rank_of: Mapping[bytes, int],
    *,
    cutoff: int | None = None,
) -> list[int]:
    """Apply leftmost, rank-priority BPE independently of tiktoken's encoder."""
    if cutoff is None:
        cutoff = len(tokens)
    if cutoff < 0 or cutoff > len(tokens):
        raise ValueError("cutoff must be between zero and the vocabulary size")

    try:
        parts = [rank_of[bytes((value,))] for value in piece]
    except KeyError as error:
        raise ValueError("the vocabulary does not cover all input bytes") from error

    while len(parts) >= 2:
        best_rank = cutoff
        best_position = -1

        for position in range(len(parts) - 1):
            left = tokens[parts[position]]
            right = tokens[parts[position + 1]]
            if left is None or right is None:
                raise ValueError("a mergeable rank resolved to a reserved token")
            candidate = rank_of.get(left + right)
            # Strict '<' deliberately preserves the leftmost occurrence on a
            # rank tie.
            if candidate is not None and candidate < best_rank:
                best_rank = candidate
                best_position = position

        if best_position < 0:
            break
        parts[best_position : best_position + 2] = [best_rank]

    return parts


def recover_merge_parents(
    tokens: Sequence[bytes | None],
    rank_of: Mapping[bytes, int],
    *,
    base_token_count: int = BASE_TOKEN_COUNT,
) -> np.ndarray:
    """Recover parents[w] = (u, v) using only ranks lower than w."""
    if not 0 < base_token_count <= len(tokens):
        raise ValueError("invalid base token count")

    parents = np.full((len(tokens), 2), -1, dtype=np.int32)
    for child in range(base_token_count, len(tokens)):
        token = tokens[child]
        if token is None:
            continue
        pieces = reference_bpe_ids(token, tokens, rank_of, cutoff=child)
        if len(pieces) != 2:
            raise ValueError(
                f"rank {child} does not reduce to exactly two parents: {pieces}"
            )
        left, right = pieces
        if not (left < child and right < child):
            raise ValueError(f"rank {child} has non-prior parents {(left, right)}")
        left_token = tokens[left]
        right_token = tokens[right]
        if left_token is None or right_token is None:
            raise ValueError(f"rank {child} refers to a reserved parent")
        if left_token + right_token != token:
            raise ValueError(f"rank {child} parents do not reconstruct the token")
        parents[child] = (left, right)

    return parents
