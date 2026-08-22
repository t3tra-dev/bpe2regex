"""Measure the experimental r50k canonical-token control-flow compiler."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from bpe2regex import CanonicalRegexBPE
from bpe2regex.encoding import Encoding
from bpe2regex.reir import (
    CanonicalTokenDFACompiler,
    CanonicalTokenRegexCompiler,
    minimize_dfa,
    prune_dead_states,
    raw_deflate_size,
)
from bpe2regex.reir.emitter.python import emit_sources
from bpe2regex.vocabulary import (
    load_vocabulary,
    recover_merge_parents,
    reference_bpe_ids,
)


def _limits(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "limits must be comma-separated integers"
        ) from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("limits must contain non-negative integers")
    return values


def _validation_words(
    tokens: tuple[bytes | None, ...],
    active_token_count: int,
    count: int,
) -> tuple[bytes, ...]:
    active = tuple(token for token in tokens[:active_token_count] if token is not None)
    randomizer = random.Random(20_260_822 + active_token_count)
    words = [b"", *active[-min(len(active), 32) :]]
    for _ in range(count):
        words.append(
            b"".join(
                randomizer.choice(active) for _ in range(randomizer.randrange(1, 9))
            )
        )
    return tuple(words)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="measure canonical-token DFA and monster-regex growth for r50k",
    )
    parser.add_argument(
        "--dfa-prefixes",
        type=_limits,
        default=(20, 50, 100, 500),
        help="comma-separated merge counts (default: 20,50,100,500)",
    )
    parser.add_argument(
        "--regex-prefixes",
        type=_limits,
        default=(0, 1, 3, 5, 10, 12, 15),
        help="comma-separated merge counts (default: 0,1,3,5,10,12,15)",
    )
    parser.add_argument(
        "--validation-cases",
        type=int,
        default=50,
        help="random token-concatenation cases per regex prefix",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="tagged state-elimination beam width; 0 disables search",
    )
    return parser


def main() -> None:
    options = _parser().parse_args()
    if options.validation_cases < 0:
        raise SystemExit("--validation-cases must be non-negative")
    if options.beam_width < 0:
        raise SystemExit("--beam-width must be non-negative")

    vocabulary = load_vocabulary(Encoding.R50K)
    tokens = vocabulary.tokens
    parents = recover_merge_parents(tokens, vocabulary.rank_of)
    base_token_count = Encoding.R50K.base_token_count
    merge_count = vocabulary.mergeable_token_count - base_token_count
    dfa_compiler = CanonicalTokenDFACompiler(tokens, parents)
    regex_compiler = CanonicalTokenRegexCompiler(tokens, parents)

    dfa_observations: list[dict[str, int | float]] = []
    for limit in options.dfa_prefixes:
        print(f"DFA prefix {limit}", file=sys.stderr, flush=True)
        built = dfa_compiler.compile(merge_limit=limit)
        started = time.perf_counter()
        minimized = prune_dead_states(minimize_dfa(built.automaton).automaton).automaton
        dfa_observations.append(
            {
                "merges": limit,
                "states": built.metrics.state_count,
                "transition_groups": built.metrics.transition_group_count,
                "explicit_symbol_transitions": (
                    built.metrics.explicit_transition_count
                ),
                "construction_seconds": built.metrics.elapsed_seconds,
                "minimized_states": minimized.state_count,
                "minimized_transition_groups": minimized.transition_group_count,
                "minimized_explicit_symbol_transitions": (
                    minimized.explicit_symbol_transition_count
                ),
                "minimization_seconds": time.perf_counter() - started,
            }
        )

    regex_observations: list[dict[str, int | float | bool]] = []
    for limit in options.regex_prefixes:
        print(f"regex prefix {limit}", file=sys.stderr, flush=True)
        fixed = regex_compiler.compile_python(merge_limit=limit)
        compiled = (
            regex_compiler.compile_python(
                merge_limit=limit,
                elimination_beam_width=options.beam_width,
            )
            if options.beam_width
            else fixed
        )
        lookup = emit_sources(
            tokens[: base_token_count + limit],
            parents[: base_token_count + limit],
        )
        lookup_source = lookup.byte_to_rank + lookup.merge_pair
        regex_started = time.perf_counter()
        program = CanonicalRegexBPE(compiled)
        regex_compile_seconds = time.perf_counter() - regex_started
        words = _validation_words(
            tokens,
            base_token_count + limit,
            options.validation_cases,
        )
        validation_started = time.perf_counter()
        equivalent = True
        try:
            for word in words:
                match = program.fullmatch(word)
                expected = reference_bpe_ids(
                    word,
                    tokens,
                    vocabulary.rank_of,
                    cutoff=base_token_count + limit,
                )
                if match is None or match.token_ids != expected:
                    equivalent = False
                    break
        finally:
            program.close()
        if not equivalent:
            raise AssertionError(f"canonical regex mismatch at merge prefix {limit}")
        regex_observations.append(
            {
                "merges": limit,
                "minimized_states": compiled.ir.metrics.minimized_state_count,
                "minimized_transition_groups": (
                    compiled.ir.metrics.minimized_transition_group_count
                ),
                "fixed_elimination_seconds": fixed.ir.metrics.elimination_seconds,
                "fixed_source_bytes": len(fixed.pattern.encode("utf-8")),
                "fixed_raw_deflate_bytes": raw_deflate_size(fixed.pattern),
                "beam_width": options.beam_width,
                "explored_elimination_candidates": (
                    compiled.ir.metrics.explored_elimination_candidates
                ),
                "elimination_seconds": compiled.ir.metrics.elimination_seconds,
                "source_bytes": len(compiled.pattern.encode("utf-8")),
                "raw_deflate_bytes": raw_deflate_size(compiled.pattern),
                "capture_count": len(compiled.capture_ranks),
                "regex_compile_seconds": regex_compile_seconds,
                "validation_cases": len(words),
                "validation_seconds": time.perf_counter() - validation_started,
                "equivalent": equivalent,
                "lookup_source_bytes": len(lookup_source.encode("utf-8")),
                "lookup_raw_deflate_bytes": raw_deflate_size(lookup_source),
            }
        )

    projection: dict[str, int | float] = {}
    if dfa_observations:
        last_dfa = dfa_observations[-1]
        observed_merges = int(last_dfa["merges"])
        if observed_merges:
            projection["quadratic_transition_groups"] = round(
                int(last_dfa["transition_groups"])
                * (merge_count / observed_merges) ** 2
            )
    if regex_observations and int(regex_observations[0]["merges"]) == 0:
        base = regex_observations[0]
        last_regex = regex_observations[-1]
        observed_merges = int(last_regex["merges"])
        if observed_merges:
            for name in ("source_bytes", "raw_deflate_bytes"):
                projection[f"linear_{name}"] = round(
                    int(base[name])
                    + (int(last_regex[name]) - int(base[name]))
                    * merge_count
                    / observed_merges
                )

    artifact = Path(".artifacts/r50k/python.bin")
    result = {
        "encoding": Encoding.R50K.value,
        "merge_count": merge_count,
        "dfa": dfa_observations,
        "regex": regex_observations,
        "naive_projection_not_forecast": projection,
        "current_python_artifact_bytes": (
            artifact.stat().st_size if artifact.exists() else None
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
