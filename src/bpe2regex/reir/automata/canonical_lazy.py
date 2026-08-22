"""Lazy residual quotient for persistent canonical-token adjacency."""

import bisect
import time
from dataclasses import dataclass, field

from .canonical_adjacency import CanonicalAdjacencyIR
from .ir import DFA, Transition
from .labels import SymbolSet


@dataclass(frozen=True, order=True, slots=True)
class TokenCone:
    """One token plus full clone subtrees after a canonical child cutoff."""

    root: int
    child_cutoff: int


@dataclass(frozen=True, slots=True)
class DeniedTokenSignature:
    """Canonical antichain whose union is one final denied-token set."""

    cones: tuple[TokenCone, ...] = ()


class TokenCloneIndex:
    """Canonical set algebra over the active token-clone forest."""

    def __init__(self, adjacency: CanonicalAdjacencyIR) -> None:
        self.adjacency = adjacency
        children: dict[int, list[int]] = {}
        for token in adjacency.active_tokens:
            parent = adjacency.token_parents[token]
            if parent is not None:
                children.setdefault(parent, []).append(token)
        for row in children.values():
            row.sort(key=lambda token: adjacency.token_births[token])
        self.children = {parent: tuple(row) for parent, row in children.items()}
        self.child_births = {
            parent: tuple(adjacency.token_births[token] for token in row)
            for parent, row in self.children.items()
        }
        self.child_positions = {
            token: position
            for row in self.children.values()
            for position, token in enumerate(row)
        }

    def cone(self, root: int, threshold: int) -> TokenCone:
        if not 0 <= root < self.adjacency.alphabet_size:
            raise ValueError("a token cone root is outside the alphabet")
        if self.adjacency.token_births[root] < 0:
            raise ValueError("a token cone root must be active")
        return TokenCone(
            root,
            bisect.bisect_right(self.child_births.get(root, ()), threshold),
        )

    def _first_child_below(self, ancestor: int, descendant: int) -> int | None:
        node = descendant
        previous = descendant
        while node != ancestor:
            previous = node
            parent = self.adjacency.token_parents[node]
            if parent is None:
                return None
            node = parent
        return previous

    def covers(self, outer: TokenCone, inner: TokenCone) -> bool:
        """Return whether every token in ``inner`` belongs to ``outer``."""
        if outer.root == inner.root:
            return outer.child_cutoff <= inner.child_cutoff
        child = self._first_child_below(outer.root, inner.root)
        return child is not None and self.child_positions[child] >= outer.child_cutoff

    def insert(
        self,
        signature: DeniedTokenSignature,
        cone: TokenCone,
    ) -> DeniedTokenSignature:
        if any(self.covers(existing, cone) for existing in signature.cones):
            return signature
        remaining = tuple(
            existing for existing in signature.cones if not self.covers(cone, existing)
        )
        return DeniedTokenSignature(tuple(sorted((*remaining, cone))))

    def denies(self, signature: DeniedTokenSignature, token: int) -> bool:
        if not 0 <= token < self.adjacency.alphabet_size:
            raise ValueError("a denied-set query token is outside the alphabet")
        if self.adjacency.token_births[token] < 0:
            return False
        for cone in signature.cones:
            if cone.root == token:
                return True
            child = self._first_child_below(cone.root, token)
            if child is not None and self.child_positions[child] >= cone.child_cutoff:
                return True
        return False


@dataclass(frozen=True, slots=True)
class LazyQuotientTransition:
    tokens: tuple[int, ...]
    target: int


@dataclass(frozen=True, slots=True)
class CanonicalLazyQuotientMetrics:
    input_state_count: int
    reachable_state_count: int
    quotient_state_count: int
    active_token_count: int
    signature_cone_count: int
    maximum_signature_cones: int
    input_dense_cell_count: int
    quotient_dense_cell_count: int
    elapsed_seconds: float


@dataclass(slots=True)
class CanonicalLazyQuotient:
    adjacency: CanonicalAdjacencyIR
    clone_index: TokenCloneIndex
    signatures: tuple[DeniedTokenSignature, ...]
    state_map: tuple[int | None, ...]
    blocks: tuple[frozenset[int], ...]
    representatives: tuple[int, ...]
    target_tokens: tuple[tuple[int, ...], ...]
    metrics: CanonicalLazyQuotientMetrics
    _row_cache: dict[int, tuple[LazyQuotientTransition, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def state_count(self) -> int:
        return len(self.blocks)

    @property
    def alphabet_size(self) -> int:
        return self.adjacency.alphabet_size

    @property
    def start(self) -> int:
        start = self.state_map[0]
        if start is None:
            raise AssertionError("the persistent start state must be reachable")
        return start

    @property
    def active_tokens(self) -> tuple[int, ...]:
        return self.adjacency.active_tokens

    def _check_state(self, state: int) -> None:
        if not 0 <= state < self.state_count:
            raise ValueError("a lazy quotient state is out of range")

    def transition(self, state: int, token: int) -> int | None:
        self._check_state(state)
        representative = self.representatives[state]
        if not self.adjacency.allowed(representative, token):
            return None
        raw_target = self.adjacency.token_targets[token]
        if raw_target is None:
            return None
        target = self.state_map[raw_target]
        if target is None:
            raise AssertionError("an active token target must be reachable")
        return target

    def transition_groups(self, state: int) -> tuple[LazyQuotientTransition, ...]:
        """Materialize one quotient row only when a consumer requests it."""
        self._check_state(state)
        known = self._row_cache.get(state)
        if known is not None:
            return known
        representative = self.representatives[state]
        transitions: list[LazyQuotientTransition] = []
        for target, tokens in enumerate(self.target_tokens):
            allowed = tuple(
                token
                for token in tokens
                if self.adjacency.allowed(representative, token)
            )
            if allowed:
                transitions.append(LazyQuotientTransition(allowed, target))
        result = tuple(transitions)
        self._row_cache[state] = result
        return result

    def to_dfa(self, *, max_cells: int | None = 10_000_000) -> DFA[bool]:
        logical_cells = self.state_count * len(self.active_tokens)
        if max_cells is not None and logical_cells > max_cells:
            raise RuntimeError(
                f"lazy quotient materialization budget exceeded: {logical_cells} cells"
            )
        rows = tuple(
            tuple(
                Transition(
                    SymbolSet.from_symbols(self.alphabet_size, transition.tokens),
                    transition.target,
                )
                for transition in self.transition_groups(state)
            )
            for state in range(self.state_count)
        )
        return DFA.accepting(
            self.alphabet_size,
            self.start,
            range(self.state_count),
            rows,
        )


class CanonicalLazyQuotientCompiler:
    """Minimize persistent 1-local adjacency without scanning its Q×Γ cells."""

    def compile(self, adjacency: CanonicalAdjacencyIR) -> CanonicalLazyQuotient:
        started = time.perf_counter()
        clone_index = TokenCloneIndex(adjacency)
        raw_signatures: list[DeniedTokenSignature] = []
        for state in range(adjacency.state_count):
            parent = adjacency.state_parents[state]
            signature = (
                DeniedTokenSignature() if parent is None else raw_signatures[parent]
            )
            for token in sorted(adjacency.state_exclusions[state]):
                signature = clone_index.insert(
                    signature,
                    clone_index.cone(token, adjacency.state_births[state]),
                )
            raw_signatures.append(signature)

        reachable: list[int] = [0]
        seen = {0}
        for token in adjacency.active_tokens:
            target = adjacency.token_targets[token]
            if target is not None and target not in seen:
                seen.add(target)
                reachable.append(target)

        signature_blocks: dict[DeniedTokenSignature, int] = {}
        block_members: list[set[int]] = []
        representatives: list[int] = []
        state_map: list[int | None] = [None] * adjacency.state_count
        for state in reachable:
            signature = raw_signatures[state]
            block = signature_blocks.get(signature)
            if block is None:
                block = len(block_members)
                signature_blocks[signature] = block
                block_members.append(set())
                representatives.append(state)
            block_members[block].add(state)
            state_map[state] = block

        target_tokens: list[list[int]] = [[] for _ in block_members]
        for token in adjacency.active_tokens:
            raw_target = adjacency.token_targets[token]
            if raw_target is None:
                raise AssertionError("an active token must have a target")
            target = state_map[raw_target]
            if target is None:
                raise AssertionError("an active token target must be reachable")
            target_tokens[target].append(token)

        signatures = tuple(raw_signatures[state] for state in representatives)
        cone_count = sum(len(signature.cones) for signature in signatures)
        metrics = CanonicalLazyQuotientMetrics(
            adjacency.state_count,
            len(reachable),
            len(block_members),
            len(adjacency.active_tokens),
            cone_count,
            max((len(signature.cones) for signature in signatures), default=0),
            adjacency.state_count * len(adjacency.active_tokens),
            len(block_members) * len(adjacency.active_tokens),
            time.perf_counter() - started,
        )
        return CanonicalLazyQuotient(
            adjacency,
            clone_index,
            signatures,
            tuple(state_map),
            tuple(frozenset(members) for members in block_members),
            tuple(representatives),
            tuple(tuple(tokens) for tokens in target_tokens),
            metrics,
        )


__all__ = [
    "CanonicalLazyQuotient",
    "CanonicalLazyQuotientCompiler",
    "CanonicalLazyQuotientMetrics",
    "DeniedTokenSignature",
    "LazyQuotientTransition",
    "TokenCloneIndex",
    "TokenCone",
]
