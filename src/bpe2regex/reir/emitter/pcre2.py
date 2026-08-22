"""Experimental PCRE2 emitter that preserves shared REIR subgraphs."""

from dataclasses import dataclass

from ..marked import Boundary, marker_count
from ..ops import Alternate, CharSet, Concat, Epsilon, Literal, Never, Op, Repeat


@dataclass(frozen=True, slots=True)
class PCRE2DAGSource:
    pattern: str
    definition_count: int
    shared_operation_count: int
    boundary_capture_count: int
    expanded_source_bytes: int
    source_bytes: int

    @property
    def saved_source_bytes(self) -> int:
        return self.expanded_source_bytes - self.source_bytes

    @property
    def compression_ratio(self) -> float:
        return (
            self.expanded_source_bytes / self.source_bytes if self.source_bytes else 1.0
        )

    @property
    def boundary_capture_groups(self) -> tuple[int, ...]:
        """Numeric groups carrying inline boundaries after DEFINE groups."""
        start = self.definition_count + 1
        return tuple(range(start, start + self.boundary_capture_count))


def _byte_escape(byte: int) -> str:
    return f"\\x{byte:02x}"


def _charset_source(op: CharSet) -> str:
    if op.bits.bit_count() == 1:
        return _byte_escape(next(iter(op.symbols)))
    fragments: list[str] = []
    for start, end in op.intervals:
        start_source = _byte_escape(start)
        if start == end:
            fragments.append(start_source)
            continue
        end_source = _byte_escape(end)
        ranged = f"{start_source}-{end_source}"
        expanded = "".join(_byte_escape(symbol) for symbol in range(start, end + 1))
        fragments.append(ranged if len(ranged) < len(expanded) else expanded)
    return "[" + "".join(fragments) + "]"


def _quantifier(op: Repeat) -> str:
    match op.min, op.max:
        case 0, 1:
            return "?"
        case 0, None:
            return "*"
        case 1, None:
            return "+"
        case minimum, maximum if minimum == maximum:
            return f"{{{minimum}}}"
        case minimum, None:
            return f"{{{minimum},}}"
        case minimum, maximum:
            return f"{{{minimum},{maximum}}}"
    raise AssertionError("Repeat bounds must select a PCRE2 quantifier")


def _walk_postorder(root: Op) -> tuple[Op, ...]:
    seen: set[int] = set()
    active: set[int] = set()
    result: list[Op] = []

    def visit(op: Op) -> None:
        identity = id(op)
        if identity in seen:
            return
        if identity in active:
            raise ValueError("PCRE2 DAG emission requires an acyclic operation graph")
        active.add(identity)
        for child in op.operands:
            visit(child)
        active.remove(identity)
        seen.add(identity)
        result.append(op)

    visit(root)
    return tuple(result)


class PCRE2SubroutineDAGEmitter:
    """Lower shared REIR nodes to ``DEFINE`` blocks and named subroutine calls.

    Candidate definitions are added only when an exact source-length model says
    they reduce the complete pattern.  The model accounts for calls from the
    main expression and from every selected definition.
    """

    def __init__(self, *, max_source_bytes: int | None = None) -> None:
        if max_source_bytes is not None and max_source_bytes <= 0:
            raise ValueError("a PCRE2 source budget must be positive")
        self.max_source_bytes = max_source_bytes

    def _occurrences(
        self,
        root: Op,
        postorder: tuple[Op, ...],
    ) -> dict[int, int]:
        counts = {id(op): 0 for op in postorder}
        counts[id(root)] = 1
        for op in reversed(postorder):
            count = counts[id(op)]
            for child in op.operands:
                counts[id(child)] += count
        return counts

    def _names(self, postorder: tuple[Op, ...]) -> dict[int, str]:
        return {id(op): f"R{index}" for index, op in enumerate(postorder)}

    def _call_length(self, name: str) -> int:
        return len(f"(?&{name})")

    def _body_length(
        self,
        op: Op,
        selected: frozenset[int],
        names: dict[int, str],
        *,
        defining: int | None = None,
        memo: dict[tuple[int, int | None], int] | None = None,
    ) -> int:
        identity = id(op)
        if identity in selected and identity != defining:
            return self._call_length(names[identity])
        if isinstance(op, Boundary):
            return len("()")
        cache = {} if memo is None else memo
        key = (identity, defining)
        known = cache.get(key)
        if known is not None:
            return known
        child_lengths = tuple(
            self._body_length(
                child,
                selected,
                names,
                defining=defining,
                memo=cache,
            )
            for child in op.operands
        )
        match op:
            case Never():
                result = len("(?!)")
            case Epsilon():
                result = 0
            case CharSet():
                result = len(_charset_source(op))
            case Literal(value):
                result = len(value) * 4
            case Concat():
                result = sum(child_lengths)
            case Alternate():
                result = 4 + sum(child_lengths) + len(child_lengths) - 1
            case Repeat():
                body_length = child_lengths[0]
                grouped = not (
                    isinstance(op.body, CharSet)
                    or (isinstance(op.body, Literal) and len(op.body.value) == 1)
                    or isinstance(op.body, Alternate)
                )
                result = body_length + len(_quantifier(op)) + (4 if grouped else 0)
            case _:
                raise TypeError(f"PCRE2 emitter does not support {type(op).__name__}")
        cache[key] = result
        return result

    def _total_length(
        self,
        root: Op,
        selected: frozenset[int],
        selected_ops: dict[int, Op],
        names: dict[int, str],
    ) -> int:
        main = self._body_length(root, selected, names)
        if not selected:
            return main
        definitions = len("(?(DEFINE)") + 1
        for identity in sorted(selected, key=lambda item: names[item]):
            name = names[identity]
            body = self._body_length(
                selected_ops[identity],
                selected,
                names,
                defining=identity,
            )
            definitions += len(f"(?<{name}>)") + body
        return definitions + main

    def _select(
        self,
        root: Op,
        postorder: tuple[Op, ...],
        occurrences: dict[int, int],
        names: dict[int, str],
    ) -> frozenset[int]:
        contains_boundary: dict[int, bool] = {}
        for op in postorder:
            contains_boundary[id(op)] = isinstance(op, Boundary) or any(
                contains_boundary[id(child)] for child in op.operands
            )
        candidates = {
            id(op): op
            for op in postorder
            if op is not root
            and not isinstance(op, (Never, Epsilon, Boundary))
            and not contains_boundary[id(op)]
            and occurrences[id(op)] > 1
        }
        empty = frozenset()
        expanded_lengths = {
            identity: self._body_length(op, empty, names)
            for identity, op in candidates.items()
        }
        order = sorted(
            candidates,
            key=lambda identity: (
                -(
                    (occurrences[identity] - 1)
                    * max(
                        0,
                        expanded_lengths[identity] - self._call_length(names[identity]),
                    )
                ),
                names[identity],
            ),
        )
        selected: frozenset[int] = frozenset()
        current = self._total_length(
            root,
            selected,
            candidates,
            names,
        )
        for identity in order:
            proposed = selected | {identity}
            size = self._total_length(
                root,
                proposed,
                candidates,
                names,
            )
            if size < current:
                selected = proposed
                current = size

        changed = True
        while changed:
            changed = False
            for identity in tuple(sorted(selected, key=lambda item: names[item])):
                proposed = selected - {identity}
                size = self._total_length(
                    root,
                    proposed,
                    candidates,
                    names,
                )
                if size < current:
                    selected = proposed
                    current = size
                    changed = True
        return selected

    def _render_body(
        self,
        op: Op,
        selected: frozenset[int],
        names: dict[int, str],
        *,
        defining: int | None = None,
    ) -> str:
        identity = id(op)
        if identity in selected and identity != defining:
            return f"(?&{names[identity]})"
        if isinstance(op, Boundary):
            return "()"
        operands = tuple(
            self._render_body(
                child,
                selected,
                names,
                defining=defining,
            )
            for child in op.operands
        )
        match op:
            case Never():
                return "(?!)"
            case Epsilon():
                return ""
            case CharSet():
                return _charset_source(op)
            case Literal(value):
                return "".join(_byte_escape(byte) for byte in value)
            case Concat():
                return "".join(operands)
            case Alternate():
                return "(?:" + "|".join(operands) + ")"
            case Repeat():
                body = operands[0]
                if not (
                    isinstance(op.body, CharSet)
                    or (isinstance(op.body, Literal) and len(op.body.value) == 1)
                    or isinstance(op.body, Alternate)
                ):
                    body = f"(?:{body})"
                return body + _quantifier(op)
            case _:
                raise TypeError(f"PCRE2 emitter does not support {type(op).__name__}")

    def emit(self, root: Op) -> PCRE2DAGSource:
        postorder = _walk_postorder(root)
        boundary_fact = marker_count(root)
        has_boundary = any(isinstance(op, Boundary) for op in postorder)
        if has_boundary and not boundary_fact.is_exactly_one:
            raise ValueError("PCRE2 marked source requires exactly one Boundary")
        occurrences = self._occurrences(root, postorder)
        names = self._names(postorder)
        selected = self._select(
            root,
            postorder,
            occurrences,
            names,
        )
        selected_ops = {id(op): op for op in postorder if id(op) in selected}
        expanded = self._body_length(root, frozenset(), names)
        predicted = self._total_length(
            root,
            selected,
            selected_ops,
            names,
        )
        if self.max_source_bytes is not None and predicted > self.max_source_bytes:
            raise RuntimeError(f"PCRE2 DAG source budget exceeded: {predicted} bytes")
        boundary_capture_count = self._boundary_occurrences(root, selected)
        if len(selected) + boundary_capture_count > 65_535:
            raise RuntimeError("PCRE2 capture-group limit exceeded")
        main = self._render_body(root, selected, names)
        if selected:
            definitions = ["(?(DEFINE)"]
            for identity in sorted(selected, key=lambda item: names[item]):
                name = names[identity]
                body = self._render_body(
                    selected_ops[identity],
                    selected,
                    names,
                    defining=identity,
                )
                definitions.append(f"(?<{name}>{body})")
            definitions.append(")")
            pattern = "".join(definitions) + main
        else:
            pattern = main
        if len(pattern.encode("ascii")) != predicted:
            raise AssertionError("PCRE2 DAG length model differs from rendered source")
        return PCRE2DAGSource(
            pattern,
            len(selected),
            len(selected),
            boundary_capture_count,
            expanded,
            predicted,
        )

    def _boundary_occurrences(self, root: Op, selected: frozenset[int]) -> int:
        def visit(op: Op) -> int:
            if id(op) in selected:
                return 0
            if isinstance(op, Boundary):
                return 1
            return sum(visit(child) for child in op.operands)

        return visit(root)


def render_pcre2_dag(
    expression: Op,
    *,
    max_source_bytes: int | None = None,
) -> PCRE2DAGSource:
    return PCRE2SubroutineDAGEmitter(max_source_bytes=max_source_bytes).emit(expression)


__all__ = [
    "PCRE2DAGSource",
    "PCRE2SubroutineDAGEmitter",
    "render_pcre2_dag",
]
