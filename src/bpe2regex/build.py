import os
import platform
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .binary import (
    ECMASCRIPT_ARTIFACT_FILENAME,
    PYTHON_ARTIFACT_FILENAME,
    encode_artifact,
)
from .emitter import Compatibility, emit_regex_sources
from .emitter.ecmascript import validate_sources as validate_ecmascript_sources
from .encoding import Encoding
from .pretokenize import PreTokenizer
from .regex_program import RegexBPE
from .unicode_data import build_unicode_class_data

Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class BuildResult[EncodingT: Encoding]:
    encoding: EncodingT
    directory: Path
    metadata: dict[str, Any]


def _notify(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _replace_bytes(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary_path = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_regex_artifact[EncodingT: Encoding](
    encoding: EncodingT,
    output_directory: str | Path | None = None,
    *,
    overwrite: bool = False,
    progress: Progress | None = None,
) -> BuildResult[EncodingT]:
    """Generate self-contained Python and ECMAScript programs for an encoding."""
    import numpy as np

    from .vocabulary import (
        load_vocabulary,
        recover_merge_parents,
        vocabulary_sha256,
    )

    base_token_count = encoding.base_token_count
    started = time.perf_counter()
    directory = (
        Path(output_directory)
        if output_directory is not None
        else Path(".artifacts") / encoding.name.lower()
    )
    directory.mkdir(parents=True, exist_ok=True)
    python_target = directory / PYTHON_ARTIFACT_FILENAME
    ecmascript_target = directory / ECMASCRIPT_ARTIFACT_FILENAME
    existing = [path for path in (python_target, ecmascript_target) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"artifact files already exist: {names}")

    _notify(progress, f"loading pinned {encoding.value} vocabulary")
    vocabulary = load_vocabulary(encoding)
    _notify(
        progress,
        "recovering "
        f"{vocabulary.mergeable_token_count - base_token_count:,} merge parent pairs",
    )
    parents = recover_merge_parents(vocabulary.tokens, vocabulary.rank_of)

    _notify(progress, "compiling the Python stdlib-re program")
    python_sources = emit_regex_sources(
        vocabulary.tokens,
        parents,
        Compatibility.PYTHON,
        base_token_count=base_token_count,
    )
    program = RegexBPE(python_sources)
    try:
        for rank, token in enumerate(vocabulary.tokens[:base_token_count]):
            assert token is not None
            match = program.fullmatch(token)
            if match is None or match.token_ids != [rank]:
                raise ValueError(f"base-byte regex self-check failed for rank {rank}")
    finally:
        program.close()

    _notify(progress, "compiling ECMAScript rank-bit and merge-bucket tries")
    ecmascript_sources = emit_regex_sources(
        vocabulary.tokens,
        parents,
        Compatibility.ECMASCRIPT,
        base_token_count=base_token_count,
    )
    _notify(progress, "validating all ECMAScript base ranks and merge rules")
    validate_ecmascript_sources(ecmascript_sources, vocabulary.tokens, parents)

    _notify(progress, "embedding the pinned Unicode pre-tokenizer regex")
    unicode_data = build_unicode_class_data()
    pretokenizer = PreTokenizer(encoding, unicode_data)
    metadata: dict[str, Any] = {
        "encoding": encoding.value,
        "mergeable_token_count": vocabulary.mergeable_token_count,
        "reserved_ranks": encoding.reserved_ranks,
        "tiktoken_version": vocabulary.tiktoken_version,
        "tiktoken_source_sha256": vocabulary.source_sha256,
        "tiktoken_native_module_sha256": vocabulary.native_module_sha256,
        "vocabulary_sha256": vocabulary_sha256(vocabulary.tokens),
        "unicode_version": unicode_data["unicode_version"],
        "rust_regex_dependencies": unicode_data["rust_regex_dependencies"],
        "unicode_ranges_sha256": unicode_data["ranges_sha256"],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "python_byte_pattern_characters": len(python_sources.byte_to_rank),
        "python_merge_pattern_characters": len(python_sources.merge_pair),
        "ecmascript_byte_pattern_count": len(ecmascript_sources.byte_rank_bits),
        "ecmascript_byte_pattern_characters": sum(
            map(len, ecmascript_sources.byte_rank_bits)
        ),
        "ecmascript_merge_pattern_count": len(ecmascript_sources.merge_buckets),
        "ecmascript_merge_bucket_max_rules": ecmascript_sources.max_bucket_rules,
        "ecmascript_merge_pattern_characters": sum(
            map(len, ecmascript_sources.merge_buckets)
        ),
        "pretokenizer_pattern_characters": len(pretokenizer.source),
        "build_seconds": time.perf_counter() - started,
    }

    python_artifact = encode_artifact(
        encoding,
        Compatibility.PYTHON,
        python_sources,
        pretokenizer.source,
    )
    ecmascript_artifact = encode_artifact(
        encoding,
        Compatibility.ECMASCRIPT,
        ecmascript_sources,
        pretokenizer.source,
    )
    metadata["python_artifact_bytes"] = len(python_artifact)
    metadata["ecmascript_artifact_bytes"] = len(ecmascript_artifact)

    _notify(progress, "writing compressed Python and ECMAScript binary artifacts")
    _replace_bytes(python_target, python_artifact)
    _replace_bytes(ecmascript_target, ecmascript_artifact)
    _notify(progress, f"regex artifacts complete in {metadata['build_seconds']:.2f}s")
    return BuildResult[EncodingT](
        encoding=encoding,
        directory=directory,
        metadata=metadata,
    )
