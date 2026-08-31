"""Persistent selectors for denied token-clone cones.

The lazy quotient represents every denied-token set as an antichain of
``TokenCone`` values.  This module interns those cones once and lets equation
states refer to the shared nodes without materializing their token sets.
"""

import bisect
from collections.abc import Iterator
from dataclasses import dataclass

from ..automata.canonical_lazy import (
    CanonicalLazyQuotient,
    DeniedTokenSignature,
    TokenCone,
)


@dataclass(frozen=True, slots=True)
class DenialConeSelectorNode:
    """One shared token cone in the selector DAG."""

    root: int
    child_cutoff: int

    @property
    def cone(self) -> TokenCone:
        return TokenCone(self.root, self.child_cutoff)


@dataclass(frozen=True, slots=True)
class TokenSelector:
    """A selector root over shared cone nodes, or the complete token domain."""

    cone_ids: tuple[int, ...] = ()
    all_tokens: bool = False

    def __post_init__(self) -> None:
        if self.all_tokens and self.cone_ids:
            raise ValueError("the global token selector cannot contain cone roots")


@dataclass(frozen=True, slots=True)
class DenialConeSelectorMetrics:
    unique_cone_count: int
    selector_count: int
    cone_occurrence_count: int
    maximum_selector_cones: int


@dataclass(frozen=True, slots=True)
class PersistentDenialConeSelectorDAG:
    """Shared selector DAG for quotient-state denial signatures.

    Selector zero is the global active-token selector used by the equation
    entry point.  The remaining selectors are interned cone antichains.  Cone
    descendants stay implicit in the persistent token-clone forest, so the
    representation is proportional to unique cones rather than denied cells.
    """

    active_tokens: tuple[int, ...]
    token_parents: tuple[int | None, ...]
    children: tuple[tuple[int, ...], ...]
    child_positions: tuple[int | None, ...]
    nodes: tuple[DenialConeSelectorNode, ...]
    selectors: tuple[TokenSelector, ...]
    state_selectors: tuple[int, ...]
    metrics: DenialConeSelectorMetrics

    def __post_init__(self) -> None:
        token_count = len(self.token_parents)
        if len(self.children) != token_count or len(self.child_positions) != token_count:
            raise ValueError("selector clone-forest columns differ in length")
        if not self.selectors or not self.selectors[0].all_tokens:
            raise ValueError("selector zero must cover the complete active token domain")
        active = frozenset(self.active_tokens)
        if len(active) != len(self.active_tokens):
            raise ValueError("active selector tokens must be unique")
        if self.active_tokens != tuple(sorted(self.active_tokens)):
            raise ValueError("active selector tokens must be sorted")
        if any(not 0 <= token < token_count for token in active):
            raise ValueError("an active selector token is out of range")
        for node in self.nodes:
            if node.root not in active:
                raise ValueError("a selector cone root must be active")
            if not 0 <= node.child_cutoff <= len(self.children[node.root]):
                raise ValueError("a selector cone cutoff is out of range")
        for selector in self.selectors:
            if any(not 0 <= node < len(self.nodes) for node in selector.cone_ids):
                raise ValueError("a selector references an unknown cone node")
            if selector.cone_ids != tuple(sorted(frozenset(selector.cone_ids))):
                raise ValueError("selector cone roots must be sorted and unique")
        if any(not 0 <= selector < len(self.selectors) for selector in self.state_selectors):
            raise ValueError("a quotient state references an unknown selector")

    @property
    def global_selector(self) -> int:
        return 0

    @classmethod
    def from_quotient(
        cls,
        quotient: CanonicalLazyQuotient,
    ) -> PersistentDenialConeSelectorDAG:
        adjacency = quotient.adjacency
        active_tokens = adjacency.active_tokens
        active = frozenset(active_tokens)
        children: list[list[int]] = [[] for _ in range(adjacency.alphabet_size)]
        for token in active_tokens:
            parent = adjacency.token_parents[token]
            if parent is not None:
                children[parent].append(token)
        for row in children:
            row.sort(key=lambda token: adjacency.token_births[token])
        child_positions: list[int | None] = [None] * adjacency.alphabet_size
        for row in children:
            for position, token in enumerate(row):
                child_positions[token] = position

        unique_cones = tuple(
            sorted(
                {
                    cone
                    for signature in quotient.signatures
                    for cone in signature.cones
                }
            )
        )
        cone_ids = {cone: index for index, cone in enumerate(unique_cones)}
        selectors: list[TokenSelector] = [TokenSelector(all_tokens=True)]
        selector_ids: dict[DeniedTokenSignature, int] = {}
        state_selectors: list[int] = []
        for signature in quotient.signatures:
            selector = selector_ids.get(signature)
            if selector is None:
                selector = len(selectors)
                selector_ids[signature] = selector
                selectors.append(
                    TokenSelector(tuple(cone_ids[cone] for cone in signature.cones))
                )
            state_selectors.append(selector)

        cone_occurrences = sum(
            len(selectors[selector].cone_ids) for selector in state_selectors
        )
        metrics = DenialConeSelectorMetrics(
            len(unique_cones),
            len(selectors),
            cone_occurrences,
            max(
                (len(selectors[selector].cone_ids) for selector in state_selectors),
                default=0,
            ),
        )
        result = cls(
            active_tokens,
            adjacency.token_parents,
            tuple(tuple(row) for row in children),
            tuple(child_positions),
            tuple(
                DenialConeSelectorNode(cone.root, cone.child_cutoff)
                for cone in unique_cones
            ),
            tuple(selectors),
            tuple(state_selectors),
            metrics,
        )
        if frozenset(result.iter_tokens(result.global_selector)) != active:
            raise AssertionError("the global selector must cover every active token")
        return result

    def _check_selector(self, selector: int) -> None:
        if not 0 <= selector < len(self.selectors):
            raise ValueError("a token selector is out of range")

    def _check_token(self, token: int) -> None:
        if not 0 <= token < len(self.token_parents):
            raise ValueError("a selector token is out of range")

    def cone_contains(self, node_id: int, token: int) -> bool:
        if not 0 <= node_id < len(self.nodes):
            raise ValueError("a selector cone node is out of range")
        self._check_token(token)
        node = self.nodes[node_id]
        if token == node.root:
            return True
        descendant = token
        child = token
        while descendant != node.root:
            child = descendant
            parent = self.token_parents[descendant]
            if parent is None:
                return False
            descendant = parent
        position = self.child_positions[child]
        if position is None:
            raise AssertionError("a non-root clone descendant must have a position")
        return position >= node.child_cutoff

    def contains(self, selector: int, token: int) -> bool:
        self._check_selector(selector)
        self._check_token(token)
        root = self.selectors[selector]
        if root.all_tokens:
            position = bisect.bisect_left(self.active_tokens, token)
            return (
                position < len(self.active_tokens)
                and self.active_tokens[position] == token
            )
        return any(self.cone_contains(node, token) for node in root.cone_ids)

    def iter_cone_tokens(self, node_id: int) -> Iterator[int]:
        if not 0 <= node_id < len(self.nodes):
            raise ValueError("a selector cone node is out of range")
        node = self.nodes[node_id]
        yield node.root
        pending = list(reversed(self.children[node.root][node.child_cutoff :]))
        while pending:
            token = pending.pop()
            yield token
            pending.extend(reversed(self.children[token]))

    def iter_tokens(self, selector: int) -> Iterator[int]:
        self._check_selector(selector)
        root = self.selectors[selector]
        if root.all_tokens:
            yield from self.active_tokens
            return
        for node in root.cone_ids:
            yield from self.iter_cone_tokens(node)


__all__ = [
    "DenialConeSelectorMetrics",
    "DenialConeSelectorNode",
    "PersistentDenialConeSelectorDAG",
    "TokenSelector",
]
