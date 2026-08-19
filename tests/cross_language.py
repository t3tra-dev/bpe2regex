import json
import random
import subprocess
from pathlib import Path

import tiktoken

from bpe2regex import Encoding, load_tokenizer
from bpe2regex.vocabulary import load_vocabulary, recover_merge_parents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JAVASCRIPT_RUNNER = PROJECT_ROOT / "tests/javascript_encode.mjs"
JAVASCRIPT_MERGE_VALIDATOR = PROJECT_ROOT / "tests/javascript_validate_merges.mjs"
RANDOM_SEED = 20_260_820
RANDOM_CASE_COUNT = 1_000


def cases() -> list[str]:
    alphabet = list("abcXYZ 日本語🙂\n\r\t!/?1234567\x1c'ſßéΩД")
    randomizer = random.Random(RANDOM_SEED)
    targeted = [
        "",
        "hello world",
        "<|endoftext|>",
        "hello    world",
        "!hello1234567",
        "we'RE I'LL i'd",
        "a\ud800b",
        "a \n b",
    ]
    generated = [
        "".join(randomizer.choice(alphabet) for _ in range(randomizer.randrange(100)))
        for _ in range(RANDOM_CASE_COUNT)
    ]
    return targeted + generated


def javascript_ids(artifact: Path, texts: list[str]) -> list[list[int]]:
    payload = "".join(json.dumps(text) + "\n" for text in texts)
    result = subprocess.run(
        ["node", JAVASCRIPT_RUNNER, artifact],
        input=payload,
        text=True,
        capture_output=True,
        check=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


def validate_javascript_merges(encoding: Encoding, artifact: Path) -> int:
    vocabulary = load_vocabulary(encoding)
    parents = recover_merge_parents(vocabulary.tokens, vocabulary.rank_of)
    rules = "".join(
        f"{int(parents[child][0])},{int(parents[child][1])},{child}\n"
        for child, token in enumerate(vocabulary.tokens)
        if child >= encoding.base_token_count and token is not None
    )
    result = subprocess.run(
        ["node", JAVASCRIPT_MERGE_VALIDATOR, artifact],
        input=rules,
        text=True,
        capture_output=True,
        check=True,
    )
    return int(result.stdout)


def main() -> int:
    texts = cases()
    for encoding in Encoding:
        directory = PROJECT_ROOT / ".artifacts" / encoding.name.lower()
        python_artifact = directory / "python.bin"
        ecmascript_artifact = directory / "ecmascript.bin"
        reference = tiktoken.get_encoding(encoding.value)
        expected = [reference.encode_ordinary(text) for text in texts]

        with load_tokenizer(encoding, python_artifact) as tokenizer:
            python_actual = [tokenizer.encode_ordinary(text) for text in texts]
        if python_actual != expected:
            raise AssertionError(f"Python token IDs differ for {encoding.value}")

        node_actual = javascript_ids(ecmascript_artifact, texts)
        if node_actual != expected:
            raise AssertionError(f"Node token IDs differ for {encoding.value}")
        merge_rule_count = validate_javascript_merges(encoding, ecmascript_artifact)
        expected_rule_count = encoding.mergeable_token_count - encoding.base_token_count
        if merge_rule_count != expected_rule_count:
            raise AssertionError(
                f"Node merge rule count differs for {encoding.value}: "
                f"{merge_rule_count} != {expected_rule_count}"
            )
        print(
            f"{encoding.value}: ok ({len(texts):,} cross-language cases, "
            f"{merge_rule_count:,} V8 merge rules)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
