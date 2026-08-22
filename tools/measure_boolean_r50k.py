"""Measure transient Boolean token labels against direct trie labels on r50k."""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass

from bpe2regex import CanonicalRegexBPE
from bpe2regex.encoding import Encoding
from bpe2regex.reir import (
    CanonicalTokenRegexCompiler,
    Complement,
    Difference,
    Intersect,
    Literal,
    Op,
    Universal,
    raw_deflate_size,
)
from bpe2regex.vocabulary import (
    load_vocabulary,
    recover_merge_parents,
    reference_bpe_ids,
)


def _limits(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "prefixes must be comma-separated integers"
        ) from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("prefixes must be non-negative integers")
    return result


@dataclass(frozen=True, slots=True)
class _GraphMetrics:
    operation_occurrences: int
    unique_operations: int
    literal_byte_occurrences: int
    boolean_operation_occurrences: int
    unique_boolean_operations: int


def _graph_metrics(root: Op) -> _GraphMetrics:
    boolean_types = (Universal, Intersect, Complement, Difference)
    stack = [root]
    seen: set[int] = set()
    occurrences = 0
    literal_bytes = 0
    boolean_occurrences = 0
    unique_boolean = 0
    while stack:
        op = stack.pop()
        occurrences += 1
        literal_bytes += len(op.value) if isinstance(op, Literal) else 0
        boolean_occurrences += isinstance(op, boolean_types)
        identity = id(op)
        if identity not in seen:
            seen.add(identity)
            unique_boolean += isinstance(op, boolean_types)
        stack.extend(op.operands)
    return _GraphMetrics(
        occurrences,
        len(seen),
        literal_bytes,
        boolean_occurrences,
        unique_boolean,
    )


def _validation_words(
    tokens: tuple[bytes | None, ...],
    active_token_count: int,
    count: int,
) -> tuple[bytes, ...]:
    active = tuple(token for token in tokens[:active_token_count] if token is not None)
    randomizer = random.Random(0xB001 + active_token_count)
    words = [b"", *active[-min(len(active), 32) :]]
    for _ in range(count):
        words.append(
            b"".join(
                randomizer.choice(active) for _ in range(randomizer.randrange(1, 7))
            )
        )
    return tuple(words)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="compare direct and Boolean token labels on r50k prefixes",
    )
    parser.add_argument(
        "--prefixes",
        type=_limits,
        default=(0, 1, 3, 5, 10, 15),
        help="comma-separated merge counts (default: 0,1,3,5,10,15)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="tagged elimination beam width; 0 uses fixed SCC order",
    )
    parser.add_argument(
        "--validation-cases",
        type=int,
        default=20,
        help="random runtime checks per prefix",
    )
    return parser


def main() -> None:
    options = _parser().parse_args()
    if options.beam_width < 0:
        raise SystemExit("--beam-width must be non-negative")
    if options.validation_cases < 0:
        raise SystemExit("--validation-cases must be non-negative")

    vocabulary = load_vocabulary(Encoding.R50K)
    tokens = vocabulary.tokens
    parents = recover_merge_parents(tokens, vocabulary.rank_of)
    compiler = CanonicalTokenRegexCompiler(tokens, parents)
    observations: list[dict[str, object]] = []
    elimination = (
        {"elimination_beam_width": options.beam_width} if options.beam_width else {}
    )

    for limit in options.prefixes:
        print(f"Boolean label prefix {limit}", file=sys.stderr, flush=True)
        direct = compiler.compile_python(merge_limit=limit, **elimination)
        boolean = compiler.compile_python(
            merge_limit=limit,
            boolean_token_labels=True,
            **elimination,
        )
        symbolic = boolean.ir.symbolic_expression or boolean.ir.expression
        direct_size = len(direct.pattern.encode("utf-8"))
        boolean_size = len(boolean.pattern.encode("utf-8"))
        direct_deflate = raw_deflate_size(direct.pattern)
        boolean_deflate = raw_deflate_size(boolean.pattern)

        cutoff = Encoding.R50K.base_token_count + limit
        words = _validation_words(tokens, cutoff, options.validation_cases)
        with CanonicalRegexBPE(boolean) as program:
            for word in words:
                match = program.fullmatch(word)
                expected = reference_bpe_ids(
                    word,
                    tokens,
                    vocabulary.rank_of,
                    cutoff=cutoff,
                )
                if match is None or match.token_ids != expected:
                    raise AssertionError(
                        f"Boolean canonical regex mismatch at prefix {limit}: {word!r}"
                    )

        observations.append(
            {
                "merges": limit,
                "direct": {
                    "source_bytes": direct_size,
                    "raw_deflate_bytes": direct_deflate,
                    "graph": asdict(_graph_metrics(direct.ir.expression)),
                },
                "boolean": {
                    "source_bytes_after_core_lowering": boolean_size,
                    "raw_deflate_bytes_after_core_lowering": boolean_deflate,
                    "lowering_seconds": boolean.ir.metrics.boolean_lowering_seconds,
                    "symbolic_graph": asdict(_graph_metrics(symbolic)),
                    "lowered_graph": asdict(_graph_metrics(boolean.ir.expression)),
                },
                "delta": {
                    "source_bytes": boolean_size - direct_size,
                    "raw_deflate_bytes": boolean_deflate - direct_deflate,
                },
                "validation_cases": len(words),
                "equivalent": True,
            }
        )

    print(
        json.dumps(
            {
                "encoding": Encoding.R50K.value,
                "beam_width": options.beam_width,
                "observations": observations,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
