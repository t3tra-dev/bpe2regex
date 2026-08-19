# bpe2regex

BPE tokenizer を正規表現に変換する研究用プロジェクトです.

具体的には, `tiktoken==0.14.0` の `r50k_base` / `p50k_base` / `cl100k_base` / `o200k_base` を Python 標準 `re` または ECMAScript `RegExp` の正規表現へ変換します.

各 encoding の生成物は圧縮バイナリ 2 本だけです.

```text
.artifacts/
├─ r50k/
│  ├─ python.bin        211,297 bytes
│  └─ ecmascript.bin    221,395 bytes
├─ p50k/
│  ├─ python.bin        211,384 bytes
│  └─ ecmascript.bin    221,513 bytes
├─ cl100k/
│  ├─ python.bin        469,209 bytes
│  └─ ecmascript.bin    485,412 bytes
└─ o200k/
   ├─ python.bin        988,147 bytes
   └─ ecmascript.bin  1,014,885 bytes
```

## Binary 形式

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
capture rank count    ULEB128
capture ranks         token countから算出した固定幅little-endian整数列
...
```

format version 1 の regex は terminal ごとに匿名 capture `()`を持ち, capture index から token rank を引く side table を別領域に格納します. Python 版は base-rank regex と side table, merge-pair regex と side table, pre-tokenizer regex を格納します. ECMAScript 版は base-rank bit regex 列, merge frontier prefix 列, suffix regex 列と pattern ごとの side table, pre-tokenizer regex を格納します.

merge-pair regex の入力は rank を `0-9A-Za-z` の固定幅 base62 へ変換し, `left || right`として連結します. 現在の 4 encoding はいずれも rank 幅 3, pair 長 6 であり, 区切り文字は使用しません.

`p50k_base` の mergeable rank 空間では `50256` が special token 用に予約され, 通常 BPE token の rank は `50255` から `50257` へ飛びます. artifact と両 runtime はこの予約 rank を欠番のまま保持します.

## Encoding 種別

variant は `Encoding`, regex dialectは `Compatibility` で独立に選びます. 現在は `Encoding.R50K`, `Encoding.P50K`, `Encoding.CL100K`, `Encoding.O200K` を実装しています.

```python
type R50K = Literal[Encoding.R50K]
type P50K = Literal[Encoding.P50K]
type CL100K = Literal[Encoding.CL100K]
type O200K = Literal[Encoding.O200K]

tokenizer: Tokenizer[R50K]
result: BuildResult[R50K]
```

## Compiler IR

regex emitter は文字列へ直接 trie を書き出さず, 2 段階の中間表現を経由します.

1. `TaggedFST` は byte 列を入力, token rank を terminal 出力とする決定的・非巡回 transducer です.
2. engine 非依存の regex AST は `Literal`, `Concat`, `Alternate`, `Tag`, `Empty`, `Never` で有限写像を表現します.
3. Python / ECMAScript rendererは `Tag` を匿名captureへloweringし, capture出現順のrank side tableを同時に生成します. ECMAScriptのbase-rank bit regexでは同じFSTの出力をbit membershipへloweringします.

ECMAScript emitter は merge-pair FST の prefix-free な trie frontier をボトムアップ DP で選びます. 各 suffix regex のキャプチャ数をマージ規則数の平方根を切り上げた値以下に制限しながら, prefix・regex・side-table 境界の非圧縮 serialized cost 合計が最小になる cut を採用します. 共通 prefix は regex から除いて dispatch table へ移すため, modulo hash bucket で失われていた trie の局所性を維持できます.

## Build

CLI が artifact 生成を担当します.

```bash
uv sync
uv run bpe2regex build r50k --force
uv run bpe2regex build p50k --force
uv run bpe2regex build cl100k --force
uv run bpe2regex build o200k --force
```

Python API からも生成できます.

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

Makefile は 4 encoding の artifact を再生成してから, 両 examples でbinary 展開・regex compile・byte captures・Unicode / 境界ケースを検証します. Node.js では全 merge frontier pattern を V8 上で compile し, prefix-free 性・capture rank の欠落・重複・予約 rank 混入・pattern 幅を検査した上で, `tiktoken` から復元した全マージ親ペアを実際の prefix dispatch へ通します. 最後に決定的に生成した 1,008 入力について, `tiktoken`・Python API・Node.js APIのtoken IDが一致することを検証します.

`run` target には `ARGS` で任意の引数列を渡せます. 同じ引数が 4 encoding・両言語へ渡ります.

```bash
make -C examples run ARGS='こんにちは, 世界\!'
make -C examples run ARGS='--verify'
```

## Browser demo

[examples/web](examples/web) は外部ライブラリを使わず, ブラウザ標準の `DecompressionStream` / `RegExp` / `TextEncoder` だけで ECMAScript artifactを読み込むデモです.
artifactの取得・展開, RegExp compile, tokenize, benchmarkはWeb Worker内で直列実行します.

https://t3tra-dev.github.io/bpe2regex/

repository root を HTTP 配信してローカルで確認する場合は, artifact path を query で指定できます.

```bash
python3 -m http.server 8000
```

```text
http://localhost:8000/examples/web/?artifacts=../../.artifacts/
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
