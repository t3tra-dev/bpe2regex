from enum import Enum
from typing import Literal


class Encoding(Enum):
    """BPE encoding variants supported by the compiler."""

    R50K = "r50k_base"
    P50K = "p50k_base"
    CL100K = "cl100k_base"
    O200K = "o200k_base"

    @property
    def token_count(self) -> int:
        match self:
            case Encoding.R50K:
                return 50_256
            case Encoding.P50K:
                # Rank 50_256 belongs to <|endoftext|>, while mergeable
                # whitespace tokens continue at ranks 50_257 through 50_280.
                return 50_281
            case Encoding.CL100K:
                return 100_256
            case Encoding.O200K:
                return 199_998

    @property
    def mergeable_token_count(self) -> int:
        return self.token_count - len(self.reserved_ranks)

    @property
    def reserved_ranks(self) -> tuple[int, ...]:
        match self:
            case Encoding.P50K:
                return (50_256,)
            case _:
                return ()

    @property
    def base_token_count(self) -> int:
        return 256

    @property
    def rank_width(self) -> int:
        return len(str(self.token_count - 1))


type R50K = Literal[Encoding.R50K]
type P50K = Literal[Encoding.P50K]
type CL100K = Literal[Encoding.CL100K]
type O200K = Literal[Encoding.O200K]
