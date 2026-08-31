"""Experimental PCRE2 lowering for canonical regular equations.

This backend intentionally lives below ``equations.research``.  It requires
PCRE2 subroutine recursion, ``DEFINE``, ``\\K``, and optionally ``MARK``; it is
not consumed by the production Python or ECMAScript emitters.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from ...equations.analysis import verify_length_guarded
from ...equations.ir import (
    EQUATION_UNIVERSAL,
    EquationDifference,
    EquationTokenSelectorRef,
    RegularEquationIR,
)

type EquationPCRE2Marker = Literal["none", "boundary", "rank"]


@dataclass(slots=True)
class _TrieNode:
    children: dict[int, _TrieNode] = field(default_factory=dict)
    terminal: int | None = None


@dataclass(frozen=True, slots=True)
class EquationPCRE2Source:
    pattern: str
    marker: EquationPCRE2Marker
    state_definition_count: int
    selector_definition_count: int
    capture_group_count: int
    selector_token_occurrences: int
    source_bytes: int


class EquationPCRE2BudgetExceeded(RuntimeError):
    pass


def _byte_escape(byte: int) -> str:
    return f"\\x{byte:02x}"


def _charset_source(symbols: Iterable[int]) -> str:
    ordered = tuple(sorted(set(symbols)))
    if not ordered:
        return "(?!)"
    if len(ordered) == 1:
        return _byte_escape(ordered[0])
    intervals: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for symbol in ordered[1:]:
        if symbol == previous + 1:
            previous = symbol
            continue
        intervals.append((start, previous))
        start = previous = symbol
    intervals.append((start, previous))
    fragments: list[str] = []
    for start, end in intervals:
        if start == end:
            fragments.append(_byte_escape(start))
        else:
            ranged = f"{_byte_escape(start)}-{_byte_escape(end)}"
            expanded = "".join(_byte_escape(byte) for byte in range(start, end + 1))
            fragments.append(ranged if len(ranged) < len(expanded) else expanded)
    return "[" + "".join(fragments) + "]"


class EquationPCRE2Emitter:
    """Emit one recursive PCRE2 pattern from canonical quotient equations."""

    def __init__(
        self,
        *,
        marker: EquationPCRE2Marker = "rank",
        max_source_bytes: int | None = 16_000_000,
        max_capture_groups: int | None = 65_535,
        max_selector_token_occurrences: int | None = 25_000_000,
        disable_auto_possess: bool = True,
    ) -> None:
        if marker not in ("none", "boundary", "rank"):
            raise ValueError("unknown PCRE2 equation marker mode")
        for name, value in (
            ("max_source_bytes", max_source_bytes),
            ("max_capture_groups", max_capture_groups),
            ("max_selector_token_occurrences", max_selector_token_occurrences),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        self.marker: EquationPCRE2Marker = marker
        self.max_source_bytes = max_source_bytes
        self.max_capture_groups = max_capture_groups
        self.max_selector_token_occurrences = max_selector_token_occurrences
        self.disable_auto_possess = disable_auto_possess

    @staticmethod
    def _verify_canonical_shape(ir: RegularEquationIR) -> None:
        for equation in ir.equations:
            selector = ir.denial_selector(equation.variable)
            selector_root = ir.selector_dag.selectors[selector]
            expected = (
                EQUATION_UNIVERSAL
                if not selector_root.cone_ids
                else EquationDifference(
                    EQUATION_UNIVERSAL,
                    EquationTokenSelectorRef(selector),
                )
            )
            if equation.body != expected:
                raise ValueError(
                    "the PCRE2 research emitter only accepts canonical "
                    "quotient-shaped equations"
                )
        global_entry = EquationTokenSelectorRef(ir.selector_dag.global_selector)
        start_selector = ir.denial_selector(ir.start)
        start_root = ir.selector_dag.selectors[start_selector]
        expected_entry = (
            global_entry
            if not start_root.cone_ids
            else EquationDifference(
                global_entry,
                EquationTokenSelectorRef(start_selector),
            )
        )
        if ir.entry != expected_entry:
            raise ValueError(
                "the PCRE2 research emitter requires the canonical entry selector"
            )

    @staticmethod
    def _call(group: int) -> str:
        return f"(?{group})"

    def _selector_group(self, ir: RegularEquationIR, node: int) -> int:
        return ir.state_count + node + 1

    def _render_trie(
        self,
        ir: RegularEquationIR,
        tokens: Iterable[int],
        leaf: Callable[[int], str],
    ) -> tuple[str, int]:
        root = _TrieNode()
        count = 0
        for token in tokens:
            value = ir.tokens[token]
            if value is None or not value:
                raise ValueError("a PCRE2 selector token must be active and non-empty")
            node = root
            for byte in value:
                node = node.children.setdefault(byte, _TrieNode())
            if node.terminal is not None:
                raise ValueError("PCRE2 selector token byte strings must be unique")
            node.terminal = token
            count += 1

        def render(node: _TrieNode) -> str:
            alternatives: list[str] = []
            if node.terminal is not None:
                alternatives.append(leaf(node.terminal))
            grouped: dict[str, list[int]] = {}
            for byte, child in sorted(node.children.items()):
                grouped.setdefault(render(child), []).append(byte)
            alternatives.extend(
                _charset_source(bytes_) + suffix
                for suffix, bytes_ in grouped.items()
            )
            if not alternatives:
                return "(?!)"
            if len(alternatives) == 1:
                return alternatives[0]
            return "(?:" + "|".join(alternatives) + ")"

        return render(root), count

    def _universal_star(self, ir: RegularEquationIR) -> str:
        return (
            r"[\s\S]*"
            if len(ir.alphabet) == 256
            else _charset_source(ir.alphabet) + "*"
        )

    def emit(self, ir: RegularEquationIR) -> EquationPCRE2Source:
        verify_length_guarded(ir)
        self._verify_canonical_shape(ir)
        group_count = ir.state_count + len(ir.selector_dag.nodes)
        if self.max_capture_groups is not None and group_count > self.max_capture_groups:
            raise EquationPCRE2BudgetExceeded(
                f"PCRE2 capture-group budget exceeded: {group_count}"
            )

        universal = self._universal_star(ir)
        state_definitions: list[str] = []
        for state in range(ir.state_count):
            selector = ir.selector_dag.selectors[ir.denial_selector(state)]
            if selector.cone_ids:
                calls = "|".join(
                    self._call(self._selector_group(ir, node))
                    for node in selector.cone_ids
                )
                denied = calls if len(selector.cone_ids) == 1 else f"(?:{calls})"
                body = f"(?!{denied}\\z){universal}"
            else:
                body = universal
            state_definitions.append(f"({body})")

        selector_definitions: list[str] = []
        token_occurrences = 0
        for node in range(len(ir.selector_dag.nodes)):
            body, count = self._render_trie(
                ir,
                ir.selector_dag.iter_cone_tokens(node),
                lambda token: self._call(ir.token_target(token) + 1),
            )
            token_occurrences += count
            if (
                self.max_selector_token_occurrences is not None
                and token_occurrences > self.max_selector_token_occurrences
            ):
                raise EquationPCRE2BudgetExceeded(
                    "PCRE2 selector-token occurrence budget exceeded: "
                    f"{token_occurrences}"
                )
            selector_definitions.append(f"({body})")

        start_denied = ir.denial_selector(ir.start)

        def allowed_entry_tokens() -> Iterable[int]:
            for token in ir.active_tokens:
                if not ir.selector_dag.contains(start_denied, token):
                    yield token

        def entry_leaf(token: int) -> str:
            marker = ""
            if self.marker == "boundary":
                marker = r"\K"
            elif self.marker == "rank":
                marker = f"(*MARK:{token})\\K"
            return marker + self._call(ir.token_target(token) + 1)

        entry, entry_tokens = self._render_trie(ir, allowed_entry_tokens(), entry_leaf)
        token_occurrences += entry_tokens
        if (
            self.max_selector_token_occurrences is not None
            and token_occurrences > self.max_selector_token_occurrences
        ):
            raise EquationPCRE2BudgetExceeded(
                "PCRE2 selector-token occurrence budget exceeded: "
                f"{token_occurrences}"
            )

        prefix = "(*NO_AUTO_POSSESS)" if self.disable_auto_possess else ""
        definitions = "".join((*state_definitions, *selector_definitions))
        pattern = f"{prefix}\\G{entry}\\z(?(DEFINE){definitions})"
        source_bytes = len(pattern.encode("ascii"))
        if self.max_source_bytes is not None and source_bytes > self.max_source_bytes:
            raise EquationPCRE2BudgetExceeded(
                f"PCRE2 source budget exceeded: {source_bytes} bytes"
            )
        return EquationPCRE2Source(
            pattern,
            self.marker,
            ir.state_count,
            len(ir.selector_dag.nodes),
            group_count,
            token_occurrences,
            source_bytes,
        )


def render_equation_pcre2(
    ir: RegularEquationIR,
    *,
    marker: EquationPCRE2Marker = "rank",
    max_source_bytes: int | None = 16_000_000,
) -> EquationPCRE2Source:
    return EquationPCRE2Emitter(
        marker=marker,
        max_source_bytes=max_source_bytes,
    ).emit(ir)


__all__ = [
    "EquationPCRE2BudgetExceeded",
    "EquationPCRE2Emitter",
    "EquationPCRE2Marker",
    "EquationPCRE2Source",
    "render_equation_pcre2",
]
