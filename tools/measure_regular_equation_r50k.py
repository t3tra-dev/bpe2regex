"""Build, validate, and optionally emit the full r50k equation system."""

import argparse
import json
import random
import time
from collections.abc import Mapping, Sequence
from typing import Any

from bpe2regex import Encoding
from bpe2regex.reir import (
    CanonicalAdjacencyCompiler,
    CanonicalLazyQuotientCompiler,
    CanonicalRegularEquationCompiler,
    RegularEquationInterpreter,
    raw_deflate_size,
)
from bpe2regex.reir.equations.research.pcre2 import EquationPCRE2Emitter
from bpe2regex.vocabulary import (
    load_vocabulary,
    recover_merge_parents,
    reference_bpe_ids,
)


def _parse_prefixes(source: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in source.split(",") if item)
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("prefixes must be non-negative integers")
    return values


def _verify_prefixes(
    tokens: tuple[bytes | None, ...],
    rank_of: Mapping[bytes, int],
    parents: Any,
    prefixes: Sequence[int],
    *,
    case_count: int,
    maximum_bytes: int,
    seed: int,
) -> dict[str, Any]:
    randomizer = random.Random(seed)
    cases = tuple(
        bytes(randomizer.randrange(256) for _ in range(randomizer.randrange(1, maximum_bytes + 1)))
        for _ in range(case_count)
    )
    adjacency_compiler = CanonicalAdjacencyCompiler(tokens, parents)
    started = time.perf_counter()
    checked_words = 0
    checked_tokens = 0
    for prefix in prefixes:
        adjacency = adjacency_compiler.compile(merge_limit=prefix).adjacency
        quotient = CanonicalLazyQuotientCompiler().compile(adjacency)
        equations = CanonicalRegularEquationCompiler().compile(quotient, tokens)
        interpreter = RegularEquationInterpreter(equations.ir)
        cutoff = Encoding.R50K.base_token_count + prefix
        for word in cases:
            actual = interpreter.tokenize(word)
            expected = tuple(
                reference_bpe_ids(word, tokens, rank_of, cutoff=cutoff)
            )
            if actual != expected:
                raise AssertionError(
                    f"r50k equation mismatch at prefix {prefix}: "
                    f"{word!r}, {actual!r} != {expected!r}"
                )
            checked_words += 1
            checked_tokens += len(actual)
    return {
        "prefixes": list(prefixes),
        "case_count_per_prefix": case_count,
        "checked_words": checked_words,
        "checked_tokens": checked_tokens,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="measure the full r50k regular-equation representation"
    )
    parser.add_argument(
        "--prefixes",
        type=_parse_prefixes,
        default=(0, 1, 3, 10, 50, 100),
    )
    parser.add_argument("--random-cases", type=int, default=32)
    parser.add_argument("--maximum-bytes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0xBEE250)
    parser.add_argument("--emit-pcre2", action="store_true")
    options = parser.parse_args()
    if options.random_cases <= 0 or options.maximum_bytes <= 0:
        parser.error("random case and byte counts must be positive")

    vocabulary = load_vocabulary(Encoding.R50K)
    parents = recover_merge_parents(vocabulary.tokens, vocabulary.rank_of)
    adjacency_started = time.perf_counter()
    adjacency = CanonicalAdjacencyCompiler(
        vocabulary.tokens,
        parents,
    ).compile()
    adjacency_seconds = time.perf_counter() - adjacency_started
    quotient_started = time.perf_counter()
    quotient = CanonicalLazyQuotientCompiler().compile(adjacency.adjacency)
    quotient_seconds = time.perf_counter() - quotient_started
    equation_started = time.perf_counter()
    equations = CanonicalRegularEquationCompiler().compile(
        quotient,
        vocabulary.tokens,
    )
    equation_seconds = time.perf_counter() - equation_started

    report: dict[str, Any] = {
        "full": {
            "adjacency_seconds": adjacency_seconds,
            "quotient_seconds": quotient_seconds,
            "equation_seconds": equation_seconds,
            "states": equations.ir.state_count,
            "active_tokens": len(equations.ir.active_tokens),
            "selectors": equations.ir.selector_dag.metrics.selector_count,
            "unique_cones": equations.ir.selector_dag.metrics.unique_cone_count,
            "cone_occurrences": equations.ir.selector_dag.metrics.cone_occurrence_count,
            "minimum_guard_bytes": equations.guardedness.minimum_guard_bytes,
        },
        "randomized_prefix_validation": _verify_prefixes(
            vocabulary.tokens,
            vocabulary.rank_of,
            parents,
            options.prefixes,
            case_count=options.random_cases,
            maximum_bytes=options.maximum_bytes,
            seed=options.seed,
        ),
    }
    if options.emit_pcre2:
        emission_started = time.perf_counter()
        emitted = EquationPCRE2Emitter(max_source_bytes=None).emit(equations.ir)
        report["experimental_pcre2"] = {
            "elapsed_seconds": time.perf_counter() - emission_started,
            "source_bytes": emitted.source_bytes,
            "raw_deflate_bytes": raw_deflate_size(emitted.pattern),
            "capture_groups": emitted.capture_group_count,
            "state_definitions": emitted.state_definition_count,
            "selector_definitions": emitted.selector_definition_count,
            "selector_token_occurrences": emitted.selector_token_occurrences,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
