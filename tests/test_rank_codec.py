import unittest

from bpe2regex.encoding import Encoding
from bpe2regex.rank_codec import (
    RANK_ALPHABET,
    RANK_RADIX,
    encode_rank,
    encode_rank_pair,
    rank_code_width,
)


class RankCodecTests(unittest.TestCase):
    def test_alphabet_is_unique_and_regex_safe(self) -> None:
        self.assertEqual(RANK_RADIX, 62)
        self.assertEqual(len(set(RANK_ALPHABET)), RANK_RADIX)
        self.assertTrue(all(chr(byte).isascii() for byte in RANK_ALPHABET))
        self.assertTrue(all(chr(byte).isalnum() for byte in RANK_ALPHABET))

    def test_width_boundaries(self) -> None:
        cases = {
            1: 1,
            RANK_RADIX: 1,
            RANK_RADIX + 1: 2,
            RANK_RADIX**2: 2,
            RANK_RADIX**2 + 1: 3,
            RANK_RADIX**3: 3,
            RANK_RADIX**3 + 1: 4,
        }
        for token_count, expected in cases.items():
            with self.subTest(token_count=token_count):
                self.assertEqual(rank_code_width(token_count), expected)

    def test_known_rank_encodings(self) -> None:
        cases = {
            (0, 3): b"000",
            (9, 3): b"009",
            (10, 3): b"00A",
            (35, 3): b"00Z",
            (36, 3): b"00a",
            (61, 3): b"00z",
            (62, 3): b"010",
            (RANK_RADIX**2 - 1, 2): b"zz",
            (RANK_RADIX**2 - 1, 3): b"0zz",
            (RANK_RADIX**3 - 1, 3): b"zzz",
        }
        for (rank, width), expected in cases.items():
            with self.subTest(rank=rank, width=width):
                self.assertEqual(encode_rank(rank, width), expected)

    def test_pair_is_two_adjacent_fixed_width_fields(self) -> None:
        self.assertEqual(encode_rank_pair(61, 62, 3), b"00z010")

    def test_current_encodings_all_use_three_digits(self) -> None:
        for encoding in Encoding:
            with self.subTest(encoding=encoding):
                self.assertEqual(encoding.rank_width, 3)

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "token count"):
            rank_code_width(0)
        with self.assertRaisesRegex(ValueError, "width"):
            encode_rank(0, 0)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            encode_rank(-1, 3)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            encode_rank(RANK_RADIX**3, 3)


if __name__ == "__main__":
    unittest.main()
