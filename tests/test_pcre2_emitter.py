import shutil
import subprocess
import unittest

from bpe2regex.reir import BOUNDARY, DEFAULT_BUILDER
from bpe2regex.reir.emitter.pcre2 import (
    PCRE2SubroutineDAGEmitter,
    render_pcre2_dag,
)
from bpe2regex.reir.ops import Alternate, Concat, Literal
from bpe2regex.reir.source import RegexSourceLowerer


def _pcre2_fullmatch(source: str, subject: bytes) -> bool:
    executable = shutil.which("pcre2grep")
    if executable is None:
        raise unittest.SkipTest("pcre2grep is not installed")
    result = subprocess.run(
        (executable, "-q", "-x", source),
        input=subject + b"\n",
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return result.returncode == 0


class PCRE2SubroutineDAGEmitterTests(unittest.TestCase):
    def test_shared_subgraph_is_defined_once_and_reduces_source(self) -> None:
        shared = Alternate(
            (
                Literal(b"a-long-common-residual-x"),
                Literal(b"a-long-common-residual-y"),
            )
        )
        root = Alternate(
            tuple(
                Concat((Literal(bytes((prefix,))), shared)) for prefix in b"0123456789"
            )
        )
        expanded = RegexSourceLowerer(escape_byte=lambda byte: f"\\x{byte:02x}").lower(
            root
        )
        emitted = render_pcre2_dag(root)
        self.assertEqual(emitted.expanded_source_bytes, len(expanded.encode("ascii")))
        self.assertGreater(emitted.shared_operation_count, 0)
        self.assertLess(emitted.source_bytes, emitted.expanded_source_bytes)
        self.assertIn("(?(DEFINE)", emitted.pattern)
        self.assertIn("(?&R", emitted.pattern)

        for prefix in b"05x":
            for suffix, expected in (
                (b"a-long-common-residual-x", prefix != ord("x")),
                (b"a-long-common-residual-y", prefix != ord("x")),
                (b"not-in-the-language", False),
            ):
                self.assertEqual(
                    _pcre2_fullmatch(emitted.pattern, bytes((prefix,)) + suffix),
                    expected,
                )

    def test_boundary_stays_inline_while_pure_suffix_is_shared(self) -> None:
        shared = Alternate((Literal(b"long-tail-x"), Literal(b"long-tail-y")))
        root = Alternate(
            (
                Concat((Literal(b"a"), BOUNDARY, shared)),
                Concat((Literal(b"b"), BOUNDARY, shared)),
            )
        )
        emitted = render_pcre2_dag(root)
        self.assertEqual(emitted.boundary_capture_count, 2)
        self.assertEqual(
            emitted.boundary_capture_groups,
            tuple(
                range(
                    emitted.definition_count + 1,
                    emitted.definition_count + 3,
                )
            ),
        )
        self.assertNotIn("(?<B>", emitted.pattern)
        self.assertTrue(_pcre2_fullmatch(emitted.pattern, b"along-tail-x"))
        self.assertTrue(_pcre2_fullmatch(emitted.pattern, b"blong-tail-y"))
        self.assertFalse(_pcre2_fullmatch(emitted.pattern, b"clong-tail-x"))

    def test_non_single_boundary_language_is_rejected(self) -> None:
        expression = DEFAULT_BUILDER.repeat(BOUNDARY, 0, 1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            render_pcre2_dag(expression)

    def test_source_budget_is_checked_before_rendering(self) -> None:
        expression = DEFAULT_BUILDER.literal(b"too large")
        with self.assertRaisesRegex(RuntimeError, "budget"):
            PCRE2SubroutineDAGEmitter(max_source_bytes=1).emit(expression)


if __name__ == "__main__":
    unittest.main()
