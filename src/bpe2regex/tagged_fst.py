from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .reir.ops import Op
from .reir.tagged import TAGGED_BUILDER, tagged

type OutputLowering = Callable[[int], Op]


@dataclass(slots=True)
class _BuilderState:
    transitions: dict[int, _BuilderState] = field(default_factory=dict)
    output: int | None = None


@dataclass(frozen=True, slots=True)
class TaggedState:
    """One state in a deterministic byte-input transducer."""

    transitions: tuple[tuple[int, int], ...]
    output: int | None = None


@dataclass(frozen=True, slots=True)
class TaggedFST:
    """A deterministic acyclic transducer with integer terminal outputs."""

    states: tuple[TaggedState, ...]
    start: int = 0

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("a tagged FST must contain at least one state")
        if not 0 <= self.start < len(self.states):
            raise ValueError("the tagged FST start state is out of range")

        for state_index, state in enumerate(self.states):
            if state.output is not None and state.output < 0:
                raise ValueError(f"state {state_index} has a negative output")
            previous_symbol = -1
            for symbol, target in state.transitions:
                if not 0 <= symbol <= 0xFF:
                    raise ValueError(
                        f"state {state_index} has an invalid byte transition"
                    )
                if symbol <= previous_symbol:
                    raise ValueError(
                        f"state {state_index} transitions are not canonical"
                    )
                if not 0 <= target < len(self.states):
                    raise ValueError(
                        f"state {state_index} has an out-of-range transition"
                    )
                previous_symbol = symbol

        colors = [0] * len(self.states)

        def visit(state_index: int) -> None:
            if colors[state_index] == 1:
                raise ValueError("a tagged FST must be acyclic")
            if colors[state_index] == 2:
                return
            colors[state_index] = 1
            for _, target in self.states[state_index].transitions:
                visit(target)
            colors[state_index] = 2

        visit(self.start)
        if any(color == 0 for color in colors):
            raise ValueError("a tagged FST must not contain unreachable states")

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[bytes | bytearray | memoryview, int]],
    ) -> TaggedFST:
        root = _BuilderState()
        for key_value, output in pairs:
            key = bytes(key_value)
            if output < 0:
                raise ValueError("a tagged FST output must be non-negative")
            state = root
            for symbol in key:
                state = state.transitions.setdefault(symbol, _BuilderState())
            if state.output is not None:
                raise ValueError(
                    f"duplicate tagged FST key for outputs {state.output}, {output}"
                )
            state.output = output

        builders: list[_BuilderState] = []
        indices: dict[int, int] = {}

        def assign(state: _BuilderState) -> None:
            identity = id(state)
            if identity in indices:
                return
            indices[identity] = len(builders)
            builders.append(state)
            for _, child in sorted(state.transitions.items()):
                assign(child)

        assign(root)
        states = tuple(
            TaggedState(
                transitions=tuple(
                    (symbol, indices[id(child)])
                    for symbol, child in sorted(builder.transitions.items())
                ),
                output=builder.output,
            )
            for builder in builders
        )
        return cls(states)

    def transduce(self, value: bytes | bytearray | memoryview) -> int | None:
        state_index = self.start
        for symbol in bytes(value):
            transitions = self.states[state_index].transitions
            target = next(
                (target for candidate, target in transitions if candidate == symbol),
                None,
            )
            if target is None:
                return None
            state_index = target
        return self.states[state_index].output

    def to_regex(
        self,
        lower_output: OutputLowering = tagged,
        *,
        start: int | None = None,
    ) -> Op:
        """Lower the transducer to prefix-factored, tagged REIR."""

        start_state = self.start if start is None else start
        if not 0 <= start_state < len(self.states):
            raise ValueError("the regex start state is out of range")

        cached: dict[int, Op] = {}

        def lower_state(state_index: int) -> Op:
            known = cached.get(state_index)
            if known is not None:
                return known
            state = self.states[state_index]
            alternatives: list[Op] = []
            if state.output is not None:
                alternatives.append(lower_output(state.output))
            alternatives.extend(
                TAGGED_BUILDER.concat(
                    TAGGED_BUILDER.literal(bytes((symbol,))),
                    lower_state(target),
                )
                for symbol, target in state.transitions
            )
            result = TAGGED_BUILDER.alternate(*alternatives)
            cached[state_index] = result
            return result

        return lower_state(start_state)


__all__ = ["OutputLowering", "TaggedFST", "TaggedState"]
