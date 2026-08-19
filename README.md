# bpe2regex

BPE tokenizer を正規表現に変換する研究用プロジェクトです.

具体的には, `tiktoken==0.14.0` の `r50k_base` / `p50k_base` / `cl100k_base` / `o200k_base` を Python 標準 `re` または ECMAScript `RegExp` の正規表現へ変換します.

各 encoding の生成物は圧縮バイナリ 2 本だけです.

```text
.artifacts/
├─ r50k/
│  ├─ python.bin        292,526 bytes
│  └─ ecmascript.bin    386,247 bytes
├─ p50k/
│  ├─ python.bin        292,653 bytes
│  └─ ecmascript.bin    386,498 bytes
├─ cl100k/
│  ├─ python.bin        598,814 bytes
│  └─ ecmascript.bin    800,186 bytes
└─ o200k/
   ├─ python.bin      1,271,929 bytes
   └─ ecmascript.bin  1,690,912 bytes
```

## Binary形式

artifact 全体を raw DEFLATE で圧縮します. Python は標準 `zlib.decompress(..., wbits=-15)`, Node.js は標準 `inflateRawSync()` だけで展開できます.

展開後のコンテナは次の最小構成です:

```text
"B2RX" magic
format version        u8
encoding ID           u8
compatibility ID      u8
token count           ULEB128
base-token count      ULEB128
rank width            ULEB128
regex source          ULEB128 byte length + UTF-8
...
```

Python 版は base-rank regex, merge-pair regex, pre-tokenizer regex を格納します. ECMAScript 版は base-rank bit regex 列, データから算出した個数の merge-bucket regex 列, pre-tokenizer regex を格納します.

`p50k_base` の mergeable rank 空間では `50256` が special token 用に予約され, 通常BPE tokenのrankは `50255` から `50257` へ飛びます. artifactと両runtimeはこの予約rankを欠番のまま保持します.

## Encoding 種別

variantは `Encoding`, regex dialectは `Compatibility` で独立に選びます. 現在は `Encoding.R50K`, `Encoding.P50K`, `Encoding.CL100K`, `Encoding.O200K` を実装しています.

```python
type R50K = Literal[Encoding.R50K]
type P50K = Literal[Encoding.P50K]
type CL100K = Literal[Encoding.CL100K]
type O200K = Literal[Encoding.O200K]

tokenizer: Tokenizer[R50K]
result: BuildResult[R50K]
```

## Build

CLI が artifact 生成を担当します.

```bash
uv sync
uv run bpe2regex build r50k --force
uv run bpe2regex build p50k --force
uv run bpe2regex build cl100k --force
uv run bpe2regex build o200k --force
```

Python APIからも生成できます.

```bash
uv sync
```

```python
from bpe2regex import R50K, BuildResult, Encoding, build_regex_artifact

result: BuildResult[R50K] = build_regex_artifact(
    Encoding.R50K,
    overwrite=True,
)
```

## Examples

[python.py](examples/python.py) と [javascript.mjs](examples/javascript.mjs) はそれぞれ独立し, 1 ファイルで次を実行します:

1. `.bin` を読み込む
2. 標準モジュールのみで raw DEFLATE を展開する
3. binary container を parse する
4. regex を compile する
5. canonical BPE tokenize と Unicode pre-tokenize を実行する
6. token ごとの分かち書きを出力する

```bash
python3 examples/python.py "hello world"
node examples/javascript.mjs "hello world"
python3 examples/python.py --artifact .artifacts/o200k/python.bin "hello world"
node examples/javascript.mjs --artifact .artifacts/o200k/ecmascript.bin "hello world"
```

```text
["hello", " world"]
```

E2E 検証は [Makefile](examples/Makefile) に集約しています.

```bash
make -C examples verify
```

Makefile は4 encodingのartifactを再生成してから, 両examplesでbinary展開・regex compile・byte captures・Unicode / 境界ケースを検証します. Node.jsでは全merge bucketをV8上でcompileし, capture rankの欠落・重複・予約rank混入・bucket幅を検査した上で, `tiktoken`から復元した全merge親ペアを実際のbucket dispatchへ通します. 最後に決定的に生成した1,008入力について, `tiktoken`・Python API・Node.js APIのtoken IDが一致することを検証します.

`run` targetには `ARGS` で任意の引数列を渡せます. 同じ引数が4 encoding・両言語へ渡ります.

```bash
make -C examples run ARGS='こんにちは, 世界\!'
make -C examples run ARGS='--verify'
```

## 開発時検証

```bash
uv run ruff check src tests examples/python.py
uv run pyright
uv run python -m unittest discover -s tests -v
make -C examples verify

# 既存artifactに対するクロス言語比較だけを実行
uv run python tests/cross_language.py
```

## License

このプロジェクトは MIT License の下でライセンスされています.
