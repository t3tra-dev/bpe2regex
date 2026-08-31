import itertools
import re
import shutil
import subprocess
import unittest
from dataclasses import replace
from functools import cache

from bpe2regex.reir import (
    EQUATION_EPSILON,
    EQUATION_UNIVERSAL,
    BudgetedEquationCoreLowerer,
    CanonicalAdjacencyCompiler,
    CanonicalLazyQuotientCompiler,
    CanonicalRegularEquationCompiler,
    EquationAlternate,
    EquationConcat,
    EquationCoreLoweringBudget,
    EquationCoreLoweringBudgetExceeded,
    EquationDifference,
    EquationLiteral,
    EquationRef,
    RegularEquation,
    RegularEquationInterpreter,
    UnguardedEquationError,
    verify_length_guarded,
)
from bpe2regex.reir.equations.research.pcre2 import EquationPCRE2Emitter
from bpe2regex.reir.source import RegexSourceLowerer
from bpe2regex.vocabulary import reference_bpe_ids
from tests.test_canonical_token_dfa import PARENTS, RANK_OF, TOKENS


def _compile_equations(merge_limit: int | None = None):
    adjacency = CanonicalAdjacencyCompiler(
        TOKENS,
        PARENTS,
        base_token_count=3,
    ).compile(merge_limit=merge_limit).adjacency
    quotient = CanonicalLazyQuotientCompiler().compile(adjacency)
    result = CanonicalRegularEquationCompiler().compile(quotient, TOKENS)
    return quotient, result


def _quotient_accepts_bytes(quotient, state: int, value: bytes) -> bool:
    @cache
    def visit(current: int, position: int) -> bool:
        if position == len(value):
            return True
        for token in quotient.active_tokens:
            token_bytes = TOKENS[token]
            target = quotient.transition(current, token)
            if (
                target is not None
                and value.startswith(token_bytes, position)
                and visit(target, position + len(token_bytes))
            ):
                return True
        return False

    return visit(state, 0)


def _pcre2_matches(pattern: str, value: bytes) -> bool:
    executable = shutil.which("pcre2grep")
    if executable is None:
        raise unittest.SkipTest("pcre2grep is not installed")
    result = subprocess.run(
        (executable, "-q", pattern),
        input=value + b"\n",
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    return result.returncode == 0


def _pcre2_mark(pattern: str, value: bytes) -> int:
    executable = shutil.which("pcre2test")
    if executable is None:
        raise unittest.SkipTest("pcre2test is not installed")
    payload = (
        b"/"
        + pattern.encode("ascii")
        + b"/\n"
        + value
        + b"\\=mark\n\n"
    )
    result = subprocess.run(
        (executable, "-q"),
        input=payload,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode("utf-8", errors="replace"))
    for line in result.stdout.splitlines():
        if line.startswith(b"MK: "):
            return int(line.removeprefix(b"MK: "))
    raise AssertionError(result.stdout.decode("utf-8", errors="replace"))


class RegularEquationTests(unittest.TestCase):
    def test_equation_expression_ops_have_exact_language_semantics(self) -> None:
        _, result = _compile_equations(3)
        interpreter = RegularEquationInterpreter(result.ir)
        finite = EquationAlternate(
            (
                EQUATION_EPSILON,
                EquationConcat((EquationLiteral(b"a"), EquationLiteral(b"b"))),
            )
        )
        self.assertTrue(interpreter.matches(finite, b""))
        self.assertTrue(interpreter.matches(finite, b"ab"))
        self.assertFalse(interpreter.matches(finite, b"a"))
        without_a = EquationDifference(EQUATION_UNIVERSAL, EquationLiteral(b"a"))
        self.assertFalse(interpreter.matches(without_a, b"a"))
        self.assertTrue(interpreter.matches(without_a, b"aa"))

    def test_selector_dag_matches_every_lazy_denial_signature(self) -> None:
        quotient, result = _compile_equations()
        dag = result.ir.selector_dag
        self.assertLessEqual(
            dag.metrics.unique_cone_count,
            dag.metrics.cone_occurrence_count,
        )
        for state, signature in enumerate(quotient.signatures):
            selector = dag.state_selectors[state]
            self.assertEqual(
                frozenset(dag.iter_tokens(selector)),
                frozenset(
                    token
                    for token in quotient.active_tokens
                    if quotient.clone_index.denies(signature, token)
                ),
            )
            for token in quotient.active_tokens:
                self.assertEqual(
                    dag.contains(selector, token),
                    quotient.clone_index.denies(signature, token),
                )

    def test_all_small_words_match_every_quotient_residual(self) -> None:
        for merge_limit in (0, 1, 3, None):
            with self.subTest(merge_limit=merge_limit):
                quotient, result = _compile_equations(merge_limit)
                interpreter = RegularEquationInterpreter(result.ir)
                for state in range(quotient.state_count):
                    for length in range(5):
                        for values in itertools.product(b"abc", repeat=length):
                            word = bytes(values)
                            self.assertEqual(
                                interpreter.matches_state(state, word),
                                _quotient_accepts_bytes(quotient, state, word),
                                (merge_limit, state, word),
                            )

    def test_entry_repeatedly_selects_reference_bpe_tokens(self) -> None:
        for merge_limit in range(8):
            with self.subTest(merge_limit=merge_limit):
                _, result = _compile_equations(merge_limit)
                interpreter = RegularEquationInterpreter(result.ir)
                cutoff = 3 + merge_limit
                self.assertFalse(interpreter.matches_entry(b""))
                for length in range(1, 7):
                    for values in itertools.product(b"abc", repeat=length):
                        word = bytes(values)
                        self.assertTrue(interpreter.matches_entry(word))
                        self.assertEqual(
                            interpreter.tokenize(word),
                            tuple(
                                reference_bpe_ids(
                                    word,
                                    TOKENS,
                                    RANK_OF,
                                    cutoff=cutoff,
                                )
                            ),
                            (merge_limit, word),
                        )

    def test_length_guardedness_rejects_direct_and_boolean_references(self) -> None:
        _, result = _compile_equations(3)
        ir = result.ir
        direct = replace(
            ir,
            equations=(RegularEquation(0, EquationRef(0)), *ir.equations[1:]),
        )
        with self.assertRaises(UnguardedEquationError):
            verify_length_guarded(direct)

        boolean = replace(
            ir,
            equations=(
                RegularEquation(
                    0,
                    EquationDifference(ir.equations[0].body, EquationRef(0)),
                ),
                *ir.equations[1:],
            ),
        )
        with self.assertRaises(UnguardedEquationError):
            verify_length_guarded(boolean)

        guarded = replace(
            ir,
            equations=(
                RegularEquation(
                    0,
                    EquationConcat((EquationLiteral(b"a"), EquationRef(0))),
                ),
                *ir.equations[1:],
            ),
        )
        self.assertTrue(verify_length_guarded(guarded).is_guarded)

    def test_core_lowering_is_exact_and_stops_before_large_materialization(self) -> None:
        quotient, result = _compile_equations(3)
        lowerer = BudgetedEquationCoreLowerer()
        entry = lowerer.lower(result.ir).expression
        render = RegexSourceLowerer(escape_byte=lambda byte: f"\\x{byte:02x}")
        entry_pattern = re.compile(render.lower(entry).encode("ascii"))
        for length in range(5):
            for values in itertools.product(b"abc", repeat=length):
                word = bytes(values)
                self.assertEqual(entry_pattern.fullmatch(word) is not None, bool(word))

        denied_state = next(
            state
            for state in range(quotient.state_count)
            if result.ir.selector_dag.selectors[
                result.ir.denial_selector(state)
            ].cone_ids
        )
        residual = lowerer.lower(result.ir, state=denied_state).expression
        residual_pattern = re.compile(render.lower(residual).encode("ascii"))
        for length in range(5):
            for values in itertools.product(b"abc", repeat=length):
                word = bytes(values)
                self.assertEqual(
                    residual_pattern.fullmatch(word) is not None,
                    _quotient_accepts_bytes(quotient, denied_state, word),
                    word,
                )

        _, full = _compile_equations()
        with self.assertRaises(EquationCoreLoweringBudgetExceeded) as raised:
            BudgetedEquationCoreLowerer(
                EquationCoreLoweringBudget(max_states=1)
            ).lower(full.ir)
        self.assertEqual(raised.exception.reason, "equation state count")
        self.assertEqual(raised.exception.metrics.materialized_row_count, 0)
        self.assertEqual(raised.exception.metrics.token_checks, 0)

    def test_pcre2_backend_is_opt_in_and_matches_small_entry_language(self) -> None:
        _, result = _compile_equations(3)
        emitted = EquationPCRE2Emitter(marker="rank").emit(result.ir)
        self.assertEqual(
            emitted.capture_group_count,
            emitted.state_definition_count + emitted.selector_definition_count,
        )
        self.assertIn("(?(DEFINE)", emitted.pattern)
        self.assertIn("(*MARK:", emitted.pattern)
        self.assertIn("\\K", emitted.pattern)
        for word in (b"a", b"aaab", b"abcbc", b"cccc"):
            self.assertTrue(_pcre2_matches(emitted.pattern, word), word)
            self.assertEqual(
                _pcre2_mark(emitted.pattern, word),
                reference_bpe_ids(word, TOKENS, RANK_OF, cutoff=6)[0],
                word,
            )
        self.assertFalse(_pcre2_matches(emitted.pattern, b"d"))


if __name__ == "__main__":
    unittest.main()
