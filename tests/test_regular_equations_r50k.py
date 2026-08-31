import random
import unittest

from bpe2regex import Encoding
from bpe2regex.reir import (
    CanonicalAdjacencyCompiler,
    CanonicalLazyQuotientCompiler,
    CanonicalRegularEquationCompiler,
    RegularEquationInterpreter,
)
from bpe2regex.vocabulary import (
    load_vocabulary,
    recover_merge_parents,
    reference_bpe_ids,
)


class R50KRegularEquationPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vocabulary = load_vocabulary(Encoding.R50K)
        cls.parents = recover_merge_parents(
            cls.vocabulary.tokens,
            cls.vocabulary.rank_of,
        )

    def test_randomized_merge_prefixes_select_reference_tokens(self) -> None:
        compiler = CanonicalAdjacencyCompiler(
            self.vocabulary.tokens,
            self.parents,
            base_token_count=Encoding.R50K.base_token_count,
        )
        randomizer = random.Random(0xBEE250)
        cases = tuple(
            bytes(randomizer.randrange(256) for _ in range(randomizer.randrange(1, 49)))
            for _ in range(32)
        )
        for merge_limit in (0, 1, 3, 10, 50, 100):
            with self.subTest(merge_limit=merge_limit):
                adjacency = compiler.compile(merge_limit=merge_limit).adjacency
                quotient = CanonicalLazyQuotientCompiler().compile(adjacency)
                equations = CanonicalRegularEquationCompiler().compile(
                    quotient,
                    self.vocabulary.tokens,
                )
                interpreter = RegularEquationInterpreter(equations.ir)
                cutoff = Encoding.R50K.base_token_count + merge_limit
                for word in cases:
                    self.assertEqual(
                        interpreter.tokenize(word),
                        tuple(
                            reference_bpe_ids(
                                word,
                                self.vocabulary.tokens,
                                self.vocabulary.rank_of,
                                cutoff=cutoff,
                            )
                        ),
                        (merge_limit, word),
                    )


if __name__ == "__main__":
    unittest.main()
