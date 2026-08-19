import unittest

from bpe2regex.encoding import Encoding
from bpe2regex.pretokenize import PreTokenizer
from bpe2regex.tokenizer import repair_surrogates
from bpe2regex.unicode_data import build_unicode_class_data


class PreTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.unicode_data = build_unicode_class_data()
        cls.pretokenizer = PreTokenizer(
            Encoding.R50K,
            cls.unicode_data,
        )

    def test_pinned_range_regressions(self) -> None:
        self.assertEqual(len(self.pretokenizer.letter_ranges), 677)
        self.assertEqual(len(self.pretokenizer.number_ranges), 144)
        self.assertEqual(len(self.pretokenizer.white_space_ranges), 10)
        self.assertEqual(
            sum(end - start + 1 for start, end in self.pretokenizer.letter_ranges),
            141_028,
        )
        self.assertEqual(
            sum(end - start + 1 for start, end in self.pretokenizer.number_ranges),
            1_911,
        )
        self.assertEqual(
            sum(end - start + 1 for start, end in self.pretokenizer.white_space_ranges),
            25,
        )

    def test_targeted_splits(self) -> None:
        cases = {
            "hello123!!  ": ["hello", "123", "!!", "  "],
            "a\n\n": ["a", "\n\n"],
            "a \n b": ["a", " \n", " b"],
            "\u001ca": ["\u001c", "a"],
            "'s I'll we've": ["'s", " I", "'ll", " we", "'ve"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.pretokenizer.split(text), expected)

    def test_o200k_targeted_splits(self) -> None:
        pretokenizer = PreTokenizer(
            Encoding.O200K,
            self.unicode_data,
        )
        cases = {
            "helloWorld": ["hello", "World"],
            "HTTPServer": ["HTTPServer"],
            "we'RE I'LL i'd": ["we'RE", " I'LL", " i'd"],
            "1234567": ["123", "456", "7"],
            "a\n\n": ["a", "\n\n"],
            "a \n b": ["a", " \n", " b"],
            " 日本語とEnglish": [" 日本語とEnglish"],
            "a'ſ": ["a'ſ"],
            "a'ß": ["a", "'ß"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(pretokenizer.split(text), expected)

    def test_p50k_uses_the_r50k_split_rules(self) -> None:
        pretokenizer = PreTokenizer(
            Encoding.P50K,
            self.unicode_data,
        )
        cases = {
            "hello123!!  ": ["hello", "123", "!!", "  "],
            "a\n\n": ["a", "\n\n"],
            "a \n b": ["a", " \n", " b"],
            "'s I'll we've": ["'s", " I", "'ll", " we", "'ve"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(pretokenizer.split(text), expected)

    def test_cl100k_targeted_splits(self) -> None:
        pretokenizer = PreTokenizer(
            Encoding.CL100K,
            self.unicode_data,
        )
        cases = {
            "hello123!!  ": ["hello", "123", "!!", "  "],
            "!hello": ["!hello"],
            "1234567": ["123", "456", "7"],
            "a\n\n": ["a", "\n\n"],
            "a \n b": ["a", " \n", " b"],
            "we'RE I'LL i'd": ["we", "'RE", " I", "'LL", " i", "'d"],
            "a'ſ": ["a", "'ſ"],
            "a'ß": ["a", "'ß"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(pretokenizer.split(text), expected)

    def test_every_character_is_covered(self) -> None:
        text = "".join(
            chr(codepoint)
            for codepoint in range(0x110000)
            if not 0xD800 <= codepoint <= 0xDFFF
        )
        for encoding in Encoding:
            with self.subTest(encoding=encoding):
                pretokenizer = PreTokenizer(encoding, self.unicode_data)
                pieces = pretokenizer.split(text)
                self.assertEqual("".join(pieces), text)

    def test_surrogate_repair_matches_tiktoken_wrapper_rule(self) -> None:
        self.assertEqual(repair_surrogates("plain"), "plain")
        self.assertEqual(repair_surrogates("\ud83d\ude00"), "😀")
        self.assertEqual(repair_surrogates("a\ud800b"), "a�b")


if __name__ == "__main__":
    unittest.main()
