RANK_SEPARATOR = b","


def encode_rank(rank: int, width: int) -> bytes:
    if not 0 <= rank < 10**width:
        raise ValueError(f"rank {rank} does not fit in {width} decimal digits")
    return f"{rank:0{width}d}".encode("ascii")
