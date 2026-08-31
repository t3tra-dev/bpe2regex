"""A separate regular-equation dialect over byte languages.

Core REIR deliberately remains an acyclic seven-operation tree/DAG.  Equation
references live in this dialect so cyclic residual systems cannot accidentally
enter core canonicalization, analyses, or source emitters.
"""

from abc import ABC
from dataclasses import dataclass

from .selectors import PersistentDenialConeSelectorDAG


class EquationExpr(ABC):
    """Base class for immutable regular-equation expressions."""

    @property
    def operands(self) -> tuple[EquationExpr, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class EquationNever(EquationExpr):
    """The empty language."""


@dataclass(frozen=True, slots=True)
class EquationEpsilon(EquationExpr):
    """The language containing only the empty word."""


@dataclass(frozen=True, slots=True)
class EquationUniversal(EquationExpr):
    """All words over the module's declared byte alphabet."""


@dataclass(frozen=True, slots=True)
class EquationLiteral(EquationExpr):
    value: bytes

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("an equation literal must not be empty")


@dataclass(frozen=True, slots=True)
class EquationRef(EquationExpr):
    variable: int

    def __post_init__(self) -> None:
        if self.variable < 0:
            raise ValueError("an equation variable reference must be non-negative")


@dataclass(frozen=True, slots=True)
class EquationTokenSelectorRef(EquationExpr):
    """A selector whose token leaves continue at their global target variable."""

    selector: int

    def __post_init__(self) -> None:
        if self.selector < 0:
            raise ValueError("an equation token selector must be non-negative")


@dataclass(frozen=True, slots=True)
class EquationConcat(EquationExpr):
    parts: tuple[EquationExpr, ...]

    def __post_init__(self) -> None:
        if len(self.parts) < 2:
            raise ValueError("an equation concatenation requires at least two parts")

    @property
    def operands(self) -> tuple[EquationExpr, ...]:
        return self.parts


@dataclass(frozen=True, slots=True)
class EquationAlternate(EquationExpr):
    alternatives: tuple[EquationExpr, ...]

    def __post_init__(self) -> None:
        if len(self.alternatives) < 2:
            raise ValueError("an equation alternation requires at least two branches")

    @property
    def operands(self) -> tuple[EquationExpr, ...]:
        return self.alternatives


@dataclass(frozen=True, slots=True)
class EquationDifference(EquationExpr):
    left: EquationExpr
    right: EquationExpr

    @property
    def operands(self) -> tuple[EquationExpr, ...]:
        return self.left, self.right


type EquationTerm = (
    EquationNever
    | EquationEpsilon
    | EquationUniversal
    | EquationLiteral
    | EquationRef
    | EquationTokenSelectorRef
    | EquationConcat
    | EquationAlternate
    | EquationDifference
)


EQUATION_NEVER = EquationNever()
EQUATION_EPSILON = EquationEpsilon()
EQUATION_UNIVERSAL = EquationUniversal()


@dataclass(frozen=True, slots=True)
class RegularEquation:
    variable: int
    body: EquationExpr

    def __post_init__(self) -> None:
        if self.variable < 0:
            raise ValueError("an equation variable must be non-negative")
        if not isinstance(self.body, EquationExpr):
            raise TypeError("an equation body must be an EquationExpr")


@dataclass(frozen=True, slots=True)
class RegularEquationIR:
    """A finite, length-guarded system of byte-language equations."""

    alphabet: frozenset[int]
    start: int
    entry: EquationExpr
    equations: tuple[RegularEquation, ...]
    tokens: tuple[bytes | None, ...]
    token_targets: tuple[int | None, ...]
    selector_dag: PersistentDenialConeSelectorDAG

    def __post_init__(self) -> None:
        self.verify()

    @property
    def state_count(self) -> int:
        return len(self.equations)

    @property
    def active_tokens(self) -> tuple[int, ...]:
        return self.selector_dag.active_tokens

    def equation(self, variable: int) -> RegularEquation:
        if not 0 <= variable < self.state_count:
            raise ValueError("an equation variable is out of range")
        return self.equations[variable]

    def denial_selector(self, state: int) -> int:
        self.equation(state)
        return self.selector_dag.state_selectors[state]

    def token_target(self, token: int) -> int:
        if not 0 <= token < len(self.token_targets):
            raise ValueError("an equation token is out of range")
        target = self.token_targets[token]
        if target is None:
            raise ValueError("an inactive equation token has no target")
        return target

    def verify(self) -> None:
        if not self.alphabet:
            raise ValueError("a regular-equation module requires a non-empty alphabet")
        if any(not 0 <= symbol < 256 for symbol in self.alphabet):
            raise ValueError("a regular-equation alphabet must contain byte values")
        if not self.equations:
            raise ValueError("a regular-equation module requires at least one equation")
        if not 0 <= self.start < len(self.equations):
            raise ValueError("the regular-equation start variable is out of range")
        if tuple(equation.variable for equation in self.equations) != tuple(
            range(len(self.equations))
        ):
            raise ValueError("regular equations must be ordered by contiguous variable id")
        if len(self.tokens) != len(self.token_targets):
            raise ValueError("equation token and target columns differ in length")
        if len(self.selector_dag.token_parents) != len(self.tokens):
            raise ValueError("equation tokens and selector DAG have different domains")
        if len(self.selector_dag.state_selectors) != len(self.equations):
            raise ValueError("equation states and selector roots differ in length")

        active = frozenset(self.active_tokens)
        for token, (value, target) in enumerate(zip(self.tokens, self.token_targets)):
            if token not in active:
                if target is not None:
                    raise ValueError("an inactive equation token has a target")
                continue
            if value is None or not value:
                raise ValueError("every active equation token must contain bytes")
            if any(byte not in self.alphabet for byte in value):
                raise ValueError("an equation token is outside the declared alphabet")
            if target is None or not 0 <= target < len(self.equations):
                raise ValueError("an active equation token has an invalid target")

        def verify_expr(root: EquationExpr) -> None:
            pending = [root]
            seen: set[int] = set()
            while pending:
                expression = pending.pop()
                if not isinstance(expression, EquationExpr):
                    raise TypeError("regular-equation operands must be EquationExpr values")
                identity = id(expression)
                if identity in seen:
                    continue
                seen.add(identity)
                match expression:
                    case EquationLiteral(value):
                        if any(byte not in self.alphabet for byte in value):
                            raise ValueError(
                                "an equation literal is outside the declared alphabet"
                            )
                    case EquationRef(variable):
                        if not 0 <= variable < len(self.equations):
                            raise ValueError("an equation references an unknown variable")
                    case EquationTokenSelectorRef(selector):
                        if not 0 <= selector < len(self.selector_dag.selectors):
                            raise ValueError("an equation references an unknown selector")
                pending.extend(expression.operands)

        verify_expr(self.entry)
        for equation in self.equations:
            verify_expr(equation.body)


__all__ = [
    "EQUATION_EPSILON",
    "EQUATION_NEVER",
    "EQUATION_UNIVERSAL",
    "EquationAlternate",
    "EquationConcat",
    "EquationDifference",
    "EquationEpsilon",
    "EquationExpr",
    "EquationLiteral",
    "EquationNever",
    "EquationRef",
    "EquationTerm",
    "EquationTokenSelectorRef",
    "EquationUniversal",
    "RegularEquation",
    "RegularEquationIR",
]
