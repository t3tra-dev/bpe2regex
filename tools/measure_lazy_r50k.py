"""Measure lazy quotient/elimination against the dense r50k prefix path."""

import argparse
import json
import random
import sys
import time

from bpe2regex import BoundaryRegexBPE
from bpe2regex.encoding import Encoding
from bpe2regex.reir import (
    CanonicalAdjacencyCompiler,
    CanonicalBoundaryRegexCompiler,
    CanonicalLazyBoundaryEliminator,
    CanonicalLazyBoundaryRegexCompiler,
    CanonicalLazyQuotientCompiler,
    LazyEliminationBudgetExceeded,
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
        raise argparse.ArgumentTypeError("prefixes must contain non-negative integers")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="measure lazy canonical quotient and elimination for r50k",
    )
    parser.add_argument(
        "--prefixes",
        type=_limits,
        default=(0, 1, 3, 5, 10, 15),
        help="dense/lazy elimination prefixes (default: 0,1,3,5,10,15)",
    )
    parser.add_argument(
        "--validation-cases",
        type=int,
        default=30,
        help="random runtime comparisons per prefix",
    )
    return parser


def main() -> None:
    options = _parser().parse_args()
    if options.validation_cases < 0:
        raise SystemExit("--validation-cases must be non-negative")
    vocabulary = load_vocabulary(Encoding.R50K)
    parents = recover_merge_parents(vocabulary.tokens, vocabulary.rank_of)

    adjacency_started = time.perf_counter()
    full_adjacency = CanonicalAdjacencyCompiler(
        vocabulary.tokens,
        parents,
    ).compile()
    adjacency_seconds = time.perf_counter() - adjacency_started
    quotient_started = time.perf_counter()
    full_quotient = CanonicalLazyQuotientCompiler().compile(full_adjacency.adjacency)
    quotient_seconds = time.perf_counter() - quotient_started
    budget_reason: str | None = None
    budget_rows: int | None = None
    try:
        CanonicalLazyBoundaryEliminator().lower(full_quotient, vocabulary.tokens)
    except LazyEliminationBudgetExceeded as error:
        budget_reason = error.reason
        budget_rows = error.metrics.materialized_row_count

    dense_compiler = CanonicalBoundaryRegexCompiler(vocabulary.tokens, parents)
    lazy_compiler = CanonicalLazyBoundaryRegexCompiler(vocabulary.tokens, parents)
    observations: list[dict[str, int | float | bool]] = []
    for limit in options.prefixes:
        print(f"lazy/dense prefix {limit}", file=sys.stderr, flush=True)
        dense_started = time.perf_counter()
        dense = dense_compiler.compile_python(merge_limit=limit)
        dense_seconds = time.perf_counter() - dense_started
        lazy_started = time.perf_counter()
        lazy = lazy_compiler.compile_python(merge_limit=limit)
        lazy_seconds = time.perf_counter() - lazy_started
        if dense.boundary_pattern != lazy.boundary_pattern:
            raise AssertionError(f"source differs at merge prefix {limit}")

        active = tuple(
            token
            for token in vocabulary.tokens[: Encoding.R50K.base_token_count + limit]
            if token is not None and b"\n" not in token
        )
        randomizer = random.Random(20_260_824 + limit)
        words = [b""]
        for _ in range(options.validation_cases):
            words.append(
                b"".join(
                    randomizer.choice(active) for _ in range(randomizer.randrange(1, 7))
                )
            )
        with BoundaryRegexBPE(lazy) as program:
            equivalent = all(
                (match := program.fullmatch(word)) is not None
                and match.token_ids
                == reference_bpe_ids(
                    word,
                    vocabulary.tokens,
                    vocabulary.rank_of,
                    cutoff=Encoding.R50K.base_token_count + limit,
                )
                for word in words
            )
        if not equivalent:
            raise AssertionError(f"runtime differs at merge prefix {limit}")
        observations.append(
            {
                "merges": limit,
                "dense_input_states": dense.ir.metrics.dfa.state_count,
                "dense_minimized_states": dense.ir.metrics.minimized_state_count,
                "lazy_reachable_states": (
                    lazy.ir.metrics.quotient.reachable_state_count
                ),
                "lazy_quotient_states": lazy.ir.metrics.quotient.quotient_state_count,
                "dense_total_seconds": dense_seconds,
                "lazy_total_seconds": lazy_seconds,
                "speedup": dense_seconds / lazy_seconds,
                "lazy_adjacency_seconds": lazy.ir.metrics.adjacency.elapsed_seconds,
                "lazy_quotient_seconds": lazy.ir.metrics.quotient.elapsed_seconds,
                "lazy_elimination_seconds": (
                    lazy.ir.metrics.elimination.elapsed_seconds
                ),
                "transition_groups": (
                    lazy.ir.metrics.elimination.transition_group_count
                ),
                "source_bytes": len(lazy.boundary_pattern.encode("ascii")),
                "raw_deflate_bytes": raw_deflate_size(lazy.boundary_pattern),
                "source_identical": True,
                "runtime_equivalent": equivalent,
            }
        )

    metrics = full_quotient.metrics
    print(
        json.dumps(
            {
                "encoding": Encoding.R50K.value,
                "full": {
                    "merges": full_adjacency.metrics.applied_merges,
                    "input_states": metrics.input_state_count,
                    "reachable_states": metrics.reachable_state_count,
                    "quotient_states": metrics.quotient_state_count,
                    "active_tokens": metrics.active_token_count,
                    "signature_cones": metrics.signature_cone_count,
                    "maximum_signature_cones": metrics.maximum_signature_cones,
                    "input_dense_cells": metrics.input_dense_cell_count,
                    "quotient_dense_cells": metrics.quotient_dense_cell_count,
                    "adjacency_seconds": adjacency_seconds,
                    "quotient_seconds": quotient_seconds,
                    "default_elimination_budget_reason": budget_reason,
                    "rows_materialized_before_budget": budget_rows,
                },
                "prefixes": observations,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
