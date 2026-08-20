from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self

from .analysis import AnalysisManager
from .ops import Op


@dataclass(frozen=True, slots=True)
class LoweringContext:
    analyses: AnalysisManager


class Lowerer[ResultT](ABC):
    """Base class for a complete lowering from REIR to a target value."""

    @abstractmethod
    def lower(
        self,
        root: Op,
        *,
        analyses: AnalysisManager | None = None,
    ) -> ResultT:
        """Lower an operation graph to the target representation."""


class OpLowerer[ResultT](ABC):
    """Base class for one extensible operation-lowering rule."""

    @property
    @abstractmethod
    def op_type(self) -> type[Op]:
        """Operation class accepted by this rule."""

    @abstractmethod
    def lower(
        self,
        op: Op,
        operands: tuple[ResultT, ...],
        context: LoweringContext,
    ) -> ResultT:
        """Lower an operation after its operands have been lowered."""


type LoweringCallback[ResultT] = Callable[
    [Op, tuple[ResultT, ...], LoweringContext], ResultT
]


class FunctionalOpLowerer[ResultT](OpLowerer[ResultT]):
    def __init__(
        self,
        op_type: type[Op],
        callback: LoweringCallback[ResultT],
    ) -> None:
        self._op_type = op_type
        self.callback = callback

    @property
    def op_type(self) -> type[Op]:
        return self._op_type

    def lower(
        self,
        op: Op,
        operands: tuple[ResultT, ...],
        context: LoweringContext,
    ) -> ResultT:
        return self.callback(op, operands, context)


class RuleBasedLowerer[ResultT](Lowerer[ResultT]):
    """Recursive target lowering dispatched through a replaceable rule registry."""

    def __init__(self, rules: Iterable[OpLowerer[ResultT]] = ()) -> None:
        self._rules: dict[type[Op], OpLowerer[ResultT]] = {}
        for rule in rules:
            self.register(rule)

    @property
    def rules(self) -> tuple[OpLowerer[ResultT], ...]:
        return tuple(self._rules.values())

    def register(
        self,
        rule: OpLowerer[ResultT],
        *,
        replace: bool = False,
    ) -> Self:
        if rule.op_type in self._rules and not replace:
            raise ValueError(
                f"a lowering rule for {rule.op_type.__name__} is already registered"
            )
        self._rules[rule.op_type] = rule
        return self

    def _lookup_rule(self, op: Op) -> OpLowerer[ResultT]:
        for op_type in type(op).__mro__:
            rule = self._rules.get(op_type)
            if rule is not None:
                return rule
        raise TypeError(f"no lowering rule is registered for {type(op).__name__}")

    def lower(
        self,
        root: Op,
        *,
        analyses: AnalysisManager | None = None,
    ) -> ResultT:
        manager = AnalysisManager() if analyses is None else analyses
        context = LoweringContext(manager)
        active: set[int] = set()

        def lower_op(op: Op) -> ResultT:
            identity = id(op)
            if identity in active:
                raise ValueError("REIR lowering requires an acyclic operation graph")
            active.add(identity)
            try:
                op.verify()
                lowered_operands = tuple(lower_op(child) for child in op.operands)
                return self._lookup_rule(op).lower(op, lowered_operands, context)
            finally:
                active.remove(identity)

        return lower_op(root)


__all__ = [
    "FunctionalOpLowerer",
    "Lowerer",
    "LoweringCallback",
    "LoweringContext",
    "OpLowerer",
    "RuleBasedLowerer",
]
