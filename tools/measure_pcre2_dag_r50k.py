"""Measure PCRE2 subroutine sharing on marked canonical r50k prefixes."""

import argparse
import json
import shutil
import subprocess
import sys
import time

from bpe2regex.encoding import Encoding
from bpe2regex.reir import CanonicalBoundaryRegexCompiler, raw_deflate_size
from bpe2regex.reir.emitter.pcre2 import render_pcre2_dag
from bpe2regex.vocabulary import load_vocabulary, recover_merge_parents


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
        description="measure PCRE2 subroutine DAG source for canonical r50k",
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
        help="marked state-elimination beam width; 0 disables search",
    )
    return parser


def _pcre2_compile_check(pattern: str) -> bool | None:
    executable = shutil.which("pcre2test")
    if executable is None:
        return None
    source = f"/\\A(?:{pattern})\\z/\n\n".encode("ascii")
    result = subprocess.run(
        (executable, "-q", "-8"),
        input=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout.decode("utf-8", errors="replace")
    return result.returncode == 0 and "Failed: error" not in output


def main() -> None:
    options = _parser().parse_args()
    if options.beam_width < 0:
        raise SystemExit("--beam-width must be non-negative")
    vocabulary = load_vocabulary(Encoding.R50K)
    parents = recover_merge_parents(vocabulary.tokens, vocabulary.rank_of)
    compiler = CanonicalBoundaryRegexCompiler(vocabulary.tokens, parents)
    observations: list[dict[str, int | float | bool | None]] = []
    for limit in options.prefixes:
        print(f"PCRE2 DAG prefix {limit}", file=sys.stderr, flush=True)
        compilation_started = time.perf_counter()
        compiled = compiler.compile_python(
            merge_limit=limit,
            elimination_beam_width=(options.beam_width or None),
        )
        compilation_seconds = time.perf_counter() - compilation_started
        emission_started = time.perf_counter()
        emitted = render_pcre2_dag(compiled.ir.expression)
        emission_seconds = time.perf_counter() - emission_started
        observations.append(
            {
                "merges": limit,
                "minimized_states": compiled.ir.metrics.minimized_state_count,
                "elimination_seconds": compiled.ir.metrics.elimination_seconds,
                "compilation_seconds": compilation_seconds,
                "expanded_boundary_source_bytes": emitted.expanded_source_bytes,
                "expanded_boundary_raw_deflate_bytes": raw_deflate_size(
                    compiled.boundary_pattern
                ),
                "pcre2_dag_source_bytes": emitted.source_bytes,
                "pcre2_dag_raw_deflate_bytes": raw_deflate_size(emitted.pattern),
                "pcre2_source_reduction_ratio": emitted.compression_ratio,
                "shared_subroutine_count": emitted.shared_operation_count,
                "boundary_capture_count": emitted.boundary_capture_count,
                "token_lookup_source_bytes": len(
                    compiled.token_to_rank.encode("ascii")
                ),
                "token_lookup_capture_count": len(compiled.token_capture_ranks),
                "combined_python_source_bytes": compiled.cost.source_bytes,
                "combined_python_raw_deflate_bytes": compiled.cost.deflate_bytes,
                "combined_python_artifact_bytes": compiled.cost.artifact_bytes,
                "emission_seconds": emission_seconds,
                "pcre2_compile_check": _pcre2_compile_check(emitted.pattern),
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
