import re
import unittest

from bpe2regex.regex_ast import (
    EMPTY,
    NEVER,
    alternate,
    concat,
    literal,
    render_regex,
    tagged,
)
from bpe2regex.tagged_fst import TaggedFST, TaggedState


def _ascii_escape(byte: int) -> str:
    character = chr(byte)
    return (
        character if character.isascii() and character.isalnum() else f"\\x{byte:02x}"
    )


class RegexASTTests(unittest.TestCase):
    def test_constructors_normalize_without_losing_tags(self) -> None:
        expression = alternate(
            NEVER,
            concat(literal(b"a"), EMPTY, literal(b"b"), tagged(3)),
            concat(literal(b"c"), tagged(4)),
        )
        source = render_regex(
            expression,
            escape_byte=_ascii_escape,
            emit_tag=lambda rank: f"(?P<t{rank}>)",
        )
        self.assertEqual(source, r"(?:ab(?P<t3>)|c(?P<t4>))")

    def test_tag_requires_an_explicit_target_lowering(self) -> None:
        with self.assertRaisesRegex(ValueError, "tag emitter"):
            render_regex(tagged(1), escape_byte=_ascii_escape)


class TaggedFSTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = ((b"a", 3), (b"ab", 4), (b"b", 5), (b"cab", 6))
        self.fst = TaggedFST.from_pairs(self.pairs)

    def test_transduces_exactly_the_inserted_finite_map(self) -> None:
        for key, expected in self.pairs:
            with self.subTest(key=key):
                self.assertEqual(self.fst.transduce(key), expected)
        for key in (b"", b"aa", b"abc", b"c", b"missing"):
            with self.subTest(key=key):
                self.assertIsNone(self.fst.transduce(key))

    def test_tagged_regex_preserves_inputs_and_outputs(self) -> None:
        source = render_regex(
            self.fst.to_regex(),
            escape_byte=_ascii_escape,
            emit_tag=lambda rank: f"(?P<t{rank}>)",
        )
        pattern = re.compile(source.encode("ascii"))
        for key, expected in self.pairs:
            with self.subTest(key=key):
                match = pattern.fullmatch(key)
                self.assertIsNotNone(match)
                assert match is not None
                self.assertEqual(match.lastgroup, f"t{expected}")
        self.assertIsNone(pattern.fullmatch(b"abc"))

    def test_outputs_can_be_lowered_to_bit_membership_languages(self) -> None:
        patterns = tuple(
            re.compile(
                render_regex(
                    self.fst.to_regex(
                        lambda rank, bit=bit: EMPTY if rank & (1 << bit) else NEVER
                    ),
                    escape_byte=_ascii_escape,
                ).encode("ascii")
            )
            for bit in range(3)
        )
        for key, expected in self.pairs:
            with self.subTest(key=key):
                actual = sum(
                    1 << bit
                    for bit, pattern in enumerate(patterns)
                    if pattern.fullmatch(key) is not None
                )
                self.assertEqual(actual, expected)

    def test_regex_can_start_at_a_trie_frontier_state(self) -> None:
        frontier_state = dict(self.fst.states[self.fst.start].transitions)[ord("c")]
        source = render_regex(
            self.fst.to_regex(start=frontier_state),
            escape_byte=_ascii_escape,
            emit_tag=lambda rank: f"(?P<t{rank}>)",
        )
        match = re.fullmatch(source.encode("ascii"), b"ab")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.lastgroup, "t6")
        with self.assertRaisesRegex(ValueError, "start state"):
            self.fst.to_regex(start=len(self.fst.states))

    def test_duplicate_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate tagged FST key"):
            TaggedFST.from_pairs(((b"a", 1), (b"a", 2)))

    def test_invalid_graphs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "acyclic"):
            TaggedFST((TaggedState(((ord("a"), 0),)),))
        with self.assertRaisesRegex(ValueError, "unreachable"):
            TaggedFST((TaggedState(()), TaggedState((), output=1)))
        with self.assertRaisesRegex(ValueError, "canonical"):
            TaggedFST(
                (
                    TaggedState(((ord("b"), 1), (ord("a"), 1))),
                    TaggedState((), output=1),
                )
            )


if __name__ == "__main__":
    unittest.main()
