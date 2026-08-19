RANK_ALPHABET = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
RANK_RADIX = len(RANK_ALPHABET)


def rank_code_width(token_count: int) -> int:
    """Return the fixed base62 width required by a rank space."""
    if token_count <= 0:
        raise ValueError("token count must be positive")
    width = 1
    capacity = RANK_RADIX
    while capacity < token_count:
        width += 1
        capacity *= RANK_RADIX
    return width


def encode_rank(rank: int, width: int) -> bytes:
    """Encode one rank as a zero-padded, regex-safe base62 byte string."""
    if width <= 0:
        raise ValueError("rank code width must be positive")
    if not 0 <= rank < RANK_RADIX**width:
        raise ValueError(f"rank {rank} does not fit in {width} base62 digits")

    encoded = bytearray((RANK_ALPHABET[0],)) * width
    remainder = rank
    for position in range(width - 1, -1, -1):
        remainder, digit = divmod(remainder, RANK_RADIX)
        encoded[position] = RANK_ALPHABET[digit]
    return bytes(encoded)


def encode_rank_pair(left: int, right: int, width: int) -> bytes:
    """Encode a parent pair without a separator because both fields are fixed-width."""
    return encode_rank(left, width) + encode_rank(right, width)


__all__ = [
    "RANK_ALPHABET",
    "RANK_RADIX",
    "encode_rank",
    "encode_rank_pair",
    "rank_code_width",
]
