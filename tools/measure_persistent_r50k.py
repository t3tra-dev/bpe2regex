"""Build and measure the full persistent r50k canonical adjacency IR."""

import argparse
import json
import random
import sys
import time

from bpe2regex.encoding import Encoding
from bpe2regex.reir import CanonicalAdjacencyCompiler, CanonicalTokenDFACompiler
from bpe2regex.vocabulary import load_vocabulary, recover_merge_parents


def _limits(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "prefixes must be comma-separated integers"
        ) from error
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("prefixes must be non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="measure persistent canonical-token adjacency for full r50k",
    )
    parser.add_argument(
        "--equivalence-prefixes",
        type=_limits,
        default=(10, 100, 500),
        help="dense DFA prefixes checked cell-by-cell (default: 10,100,500)",
    )
    parser.add_argument(
        "--density-samples",
        type=int,
        default=100_000,
        help="uniform logical cells sampled for transition density",
    )
    return parser


def _max_depth(parents: tuple[int | None, ...], active: tuple[int, ...]) -> int:
    depths = [0] * len(parents)
    for item in active:
        parent = parents[item]
        depths[item] = 0 if parent is None else depths[parent] + 1
    return max((depths[item] for item in active), default=0)


def main() -> None:
    options = _parser().parse_args()
    if options.density_samples < 0:
        raise SystemExit("--density-samples must be non-negative")
    vocabulary = load_vocabulary(Encoding.R50K)
    parents_started = time.perf_counter()
    parents = recover_merge_parents(vocabulary.tokens, vocabulary.rank_of)
    parents_seconds = time.perf_counter() - parents_started
    compiler = CanonicalAdjacencyCompiler(vocabulary.tokens, parents)
    result = compiler.compile(
        checkpoint_interval=5_000,
        progress=lambda metrics: print(
            f"persistent merges={metrics.applied_merges} states={metrics.state_count}",
            file=sys.stderr,
            flush=True,
        ),
    )
    adjacency = result.adjacency

    equivalence: list[dict[str, int | float | bool]] = []
    dense_compiler = CanonicalTokenDFACompiler(vocabulary.tokens, parents)
    for limit in options.equivalence_prefixes:
        started = time.perf_counter()
        persistent = compiler.compile(merge_limit=limit).adjacency
        dense = dense_compiler.compile(merge_limit=limit).automaton
        same = persistent.state_count == dense.state_count
        checked = 0
        if same:
            for state in range(dense.state_count):
                for token in persistent.active_tokens:
                    checked += 1
                    if persistent.transition(state, token) != dense.transition(
                        state, token
                    ):
                        same = False
                        break
                if not same:
                    break
        equivalence.append(
            {
                "merges": limit,
                "checked_active_cells": checked,
                "equivalent": same,
                "seconds": time.perf_counter() - started,
            }
        )
        if not same:
            raise AssertionError(f"persistent adjacency differs at prefix {limit}")

    randomizer = random.Random(20_260_823)
    active = adjacency.active_tokens
    allowed_samples = 0
    sampling_started = time.perf_counter()
    for _ in range(options.density_samples):
        state = randomizer.randrange(adjacency.state_count)
        token = randomizer.choice(active)
        allowed_samples += adjacency.allowed(state, token)
    sampling_seconds = time.perf_counter() - sampling_started

    metrics = result.metrics
    theoretical_u32_payload = (
        metrics.state_count * 2
        + metrics.exclusion_count
        + metrics.active_token_count * 3
    ) * 4
    output = {
        "encoding": Encoding.R50K.value,
        "parents_recovery_seconds": parents_seconds,
        "full": {
            "applied_merges": metrics.applied_merges,
            "states": metrics.state_count,
            "active_tokens": metrics.active_token_count,
            "state_parent_links": metrics.state_parent_links,
            "token_parent_links": metrics.token_parent_links,
            "exclusions": metrics.exclusion_count,
            "persistent_records": metrics.persistent_record_count,
            "logical_dense_cells": metrics.dense_cell_count,
            "dense_to_persistent_record_ratio": (
                metrics.dense_cell_count / metrics.persistent_record_count
            ),
            "theoretical_u32_payload_bytes": theoretical_u32_payload,
            "state_clone_max_depth": _max_depth(
                adjacency.state_parents,
                tuple(range(adjacency.state_count)),
            ),
            "token_clone_max_depth": _max_depth(
                adjacency.token_parents,
                active,
            ),
            "construction_seconds": metrics.elapsed_seconds,
        },
        "sampled_transition_density": (
            allowed_samples / options.density_samples
            if options.density_samples
            else None
        ),
        "density_samples": options.density_samples,
        "density_sampling_seconds": sampling_seconds,
        "prefix_equivalence": equivalence,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
