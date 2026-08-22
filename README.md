# bpe2regex

BPE tokenizer を正規表現に変換する研究用プロジェクトです.

具体的には, `tiktoken==0.14.0` の `r50k_base` / `p50k_base` / `cl100k_base` / `o200k_base` を Python 標準 `re` または ECMAScript `RegExp` の正規表現へ変換します.

各 encoding の生成物は圧縮バイナリ 2 本だけです.

```text
.artifacts/
├─ r50k/
│  ├─ python.bin        211,297 bytes
│  └─ ecmascript.bin    221,079 bytes
├─ p50k/
│  ├─ python.bin        211,384 bytes
│  └─ ecmascript.bin    221,205 bytes
├─ cl100k/
│  ├─ python.bin        469,209 bytes
│  └─ ecmascript.bin    485,052 bytes
└─ o200k/
   ├─ python.bin        988,147 bytes
   └─ ecmascript.bin  1,014,543 bytes
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

regex emitter は文字列へ直接 trie を書き出さず, `bpe2regex.reir` のコンパイラインフラストラクチャを経由します.

pure REIR は byte alphabet `Σ = {0, ..., 255}` 上の次の 7 op だけで構成します.

```text
Never
Epsilon
CharSet(byte bitset)
Literal(bytes)
Concat(children...)
Alternate(children...)
Repeat(body, min, max | None)
```

`CharSet` は 256-bit の canonical bitset として保持し, source lowering 時に singleton・列挙・range の短い表現を選びます. `Alternate` は pure semantics 上で flat / sorted / unique にし, 1-byte literal と `CharSet` の union を一つの `CharSet` へまとめます. `Concat` は flat 化, `Epsilon` 除去, `Never` absorption, adjacent literal / repeat foldingを行います. `Repeat` は trivial bounds, `Epsilon` / `Never`, nested closure を fold します.

`StructureDiscoveryPass` は canonicalization と分離し, n-ary alternative の longest common prefix / suffix factoring と, contiguous な同一 expression の冪 union を bounded `Repeat` へ復元します. `RegexPropertiesAnalysis` は nullability, first / last byte set, min / max width, structural cost をボトムアップ伝播・キャッシュします.

`Tag(rank)` は pure REIR に含めず `bpe2regex.reir.tagged` の出力付き dialect に分離しています. `TaggedConcat` / `TaggedAlternate` が出力順を持つ graph を構成し, core `Concat` / `Alternate` は constructor で `PureOp` 以外の child を拒否します. tagged builder は branch order と duplicate を保持し, pure subtree だけを core builder へ委譲します. `TaggedFST` はこの dialect へ lowering され, `TaggedRegexSourceLowerer` が `Tag` を匿名 capture と capture-rank side table に変換します.

`RewritePattern` / `PatternRewriter`, `OperationPass` / `PassManager`, `Lowerer` / `OpLowerer` はすべて追加実装・登録可能です. engine 別 emitter も `bpe2regex.reir.emitter` 配下に置き, FST から source regex までを REIR コンパイラの責務としてまとめています.

`CandidateGenerator` は同値な変換候補だけを生成し, `CostModel` は候補評価を rewrite から分離します. `MinimumCostSelector` は cost model の辞書式 key が最小の候補を選び, 完全な tie では入力順で最初の候補を保持します. `CandidateSelectionPass` がこの三者を接続するため, search transformation を通常の `RegexCompiler` pipeline に追加できます.

`StructuralCostModel` は operation 数と literal byte 数, `SourceSizeCostModel` は target source の UTF-8 byte 数, `DeflatedSourceCostModel` は単独 source の raw-DEFLATE byte 数を評価します. `ArtifactSizeCostModel` は lowered output を完全な artifact bytes にする engine 固有 serializer を受け取り, container や side table も含む総 byte 数を評価します. より一般の目的関数には `FunctionalCostModel` を使用できます. `benchmark_compiler` は pass と lowering を含む時間, 最終 IR の構造, source 長, raw-DEFLATE 長を一つの結果として記録します.

`DerivativeEngine` は7つのpure opに対するBrzozowski derivativeを`(op identity, byte)`単位でmemoizeします. `group()`は`First(R)`に含まれるbyteだけを評価し, canonicalization後のresidual IRがstructurally equalなbyteを一つの`CharSet`へまとめます. DFA equivalenceによるsemantic groupingはまだ行いません.

`DerivativeFactoringGenerator` はnullabilityとgrouped derivativeから元のlanguageを再構築し, `CandidateSelectionPass`へ同値候補として渡します. original IRはselection passが候補0として保持するため, factoringは強制rewriteになりません. `SourceSizeCostModel`, `DeflatedSourceCostModel`, または完全なartifact bytesを受け取る`ArtifactSizeCostModel`で候補が勝った場合だけ採用されます.

pure REIR のproduction source loweringは`StructureDiscoveryPass`の後にこのgeneratorを実行し, artifact全体がraw DEFLATEされることに合わせて`DeflatedSourceCostModel`で選択します. tag付きlookup regexは出力順とrank semanticsを持つため, pure derivative factoringの対象には含めません.

`bpe2regex.reir.automata` は次段の control-flow compiler 用の有限 alphabet automaton IR です. `SymbolSet` は alphabet size と canonical bitset を持ち, union / intersection / difference / complement を label algebra として提供します. byte alphabet の label は pure REIR の `CharSet` へ変換できます. `DFA` は互いに素な `SymbolSet` edge を持つ immutable な partial deterministic automaton で, 未定義 transition は reject を意味します.

accept state は bool だけでなく hashable な observable output を持てます. したがって residual quotient は accept/reject が一致するだけの state を併合せず, rank 等の output まで一致する state だけを併合します. `minimize_dfa()` は到達不能 state の除去, 暗黙 reject sink による totalization, global symbol equivalence class の構築, output-aware Hopcroft refinement, BFS 順の canonical reindex を行い, 元 state から quotient state への map も返します. `equivalence_counterexample()` は二つの partial DFA に対し, output が異なる最短かつ辞書順最小の入力列を返します.

`DefaultTransition` は明示 edge が覆わない残余 alphabet を一つの target へ送る構文です. `encode_default_transitions()` は total row で最大の symbol set を default 化し, `expand_default_transitions()` / `effective_transitions()` はそれを厳密な補集合 label へ戻します. partial row は未定義入力が reject であるため自動 totalize しません.

`ArdenEliminator` は対象 output へ到達できない state を除き, condensation DAG の下流 SCC から GNFA state を消去して pure REIR を生成します. 各更新は loop を `Repeat(loop, 0, None)` とした Arden の `prefix · loop* · suffix` です. 全 accepting state の union に加え, observable output ごとの言語も個別に lower できます.

`CostGuidedArdenEliminator` は SCC 間の順序を固定したまま SCC 内の state-elimination 順序を beam search します. partial graph は任意の `CostModel` で評価し, 最終候補は完成した REIR の source / DEFLATE / artifact cost で比較できます. beam が狭くても決定的な SCC baseline は必ず最終候補に残します.

`AutomatonSemanticAbsorber` は pure acceptance DFA の union に対し積 automaton で言語包含を証明し, `L(A) ⊆ L(B)` の alternative `A` を除去します. 同値な alternative は入力順の先頭だけを残します. rank/tag output を持つ DFA にはこの absorption を適用しません.

`AcceptanceAutomataCompiler` は pure DFA 群に residual minimization, semantic absorption, default-transition encoding, Arden elimination を順に適用し, union 全体を一つの REIR へ lower します. eliminator は `CostGuidedArdenEliminator` へ差し替え可能です.

`CanonicalTokenDFACompiler` は byte-token の universal DFA に rank 順で merge rule を適用し, canonical token 列だけを受理する token-alphabet DFA を構築します. incremental construction は [Constructing a BPE Tokenization DFA](https://arxiv.org/abs/2405.07671) の Algorithm 2 を byte vocabulary 全体へ適用したものです. 構築途中の state / transition-group budget と checkpoint callback を持つため, 表現爆発を bounded に観測できます. その後 `minimize_dfa()` と `prune_dead_states()` で residual quotient と totalization 時の reject sink を除きます.

`TokenSymbolLowerer` は token-rank の `SymbolSet` を token bytes の prefix-factored pure REIR へ変換します. `ArdenEliminator` は byte alphabet 固定ではなく任意の label lowerer と開始 state を受け取れるため, token DFA の residual language も同じ基盤で処理できます. `CanonicalTokenRegexCompiler` は開始 edge だけを `Literal(token) · Tag(rank)` とし, 残りを pure token language として SCC-aware elimination した tagged REIR / Python regex source を生成します.

`CanonicalEliminationOrderSearcher` は SCC 間の順序を保ち, SCC 内の tagged GNFA elimination 順序を beam 探索します. partial candidate は operation occurrence / literal byte の cheap cost で prune し, 完成候補は Python source の raw-DEFLATE / source byte 数で選びます. `compile_python(elimination_beam_width=3)` のように opt-in でき, 固定 SCC 順も候補0として必ず残します.

これは実験 API として接続済みですが, `.artifacts` の format version 1 と既定の `RegexBPE` は引き続き lookup regex + heap BPE です. full r50k canonical regex は現在の elimination では生成不能な大きさになるため, production artifact への切替はまだ行っていません.

### Canonical tokenizer の matching 契約

canonical-token compiler が生成する monster regex は, 一回の `fullmatch` から全 token 境界を返すものとはしません. pre-tokenizer が生成した各 piece に対し, 同一 pattern を現在位置から piece の末尾まで `fullmatch` します. accepting path で参加する一つの空 capture は先頭 canonical token の直後にあり, driver は match 全体の終端ではなくその capture 位置へ進めます. 残り suffix に対してこれを繰り返すことで全 token rank と境界を送出します.

`CanonicalRegexBPE` の driver に残る control flow は fullmatch の反復, capture-rank side table の参照, 結果の送出だけです. merge-pair lookup, rank priority queue, merge rule の適用判断は regex 側へ compile されます. 各 capture 境界は必ず一 byte 以上進みます.

### r50k prefix での効果と爆発

2026-08-22 に current implementation で測った決定的な source-size 結果です. `lookup` は同じ merge prefix の既存 Python lookup regex 2 本の連結, `monster` は control flow を含む一つの canonical regex です. DEFLATE は raw stream 単独の byte 数です.

| merges | DFA states / groups | minimized states / groups | monster source | monster DEFLATE | lookup source | lookup DEFLATE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 / 1 | 1 / 1 | 1,807 | 442 | 1,799 | 423 |
| 1 | 2 / 4 | 2 / 4 | 3,857 | 524 | 1,801 | 426 |
| 3 | 4 / 12 | 3 / 9 | 9,195 | 654 | 1,820 | 445 |
| 5 | 6 / 30 | 4 / 16 | 26,075 | 893 | 1,836 | 460 |
| 10 | 11 / 95 | 7 / 47 | 1,112,387 | 12,528 | 1,865 | 478 |
| 12 | 13 / 111 | 7 / 47 | 1,146,075 | 13,043 | 1,873 | 483 |
| 15 | 16 / 137 | 8 / 62 | 5,658,807 | 51,487 | 1,893 | 495 |

同じ DFA に beam width 3 の elimination 順序探索を接続した結果は次の通りです.

| merges | fixed source / DEFLATE | searched source / DEFLATE | searched captures |
|---:|---:|---:|---:|
| 0 | 1,807 / 442 | 1,807 / 442 | 256 |
| 1 | 3,857 / 524 | 1,961 / 500 | 258 |
| 3 | 9,195 / 654 | 2,510 / 582 | 263 |
| 5 | 26,075 / 893 | 4,104 / 679 | 273 |
| 10 | 1,112,387 / 12,528 | 48,951 / 1,359 | 395 |
| 12 | 1,146,075 / 13,043 | 53,558 / 1,416 | 404 |
| 15 | 5,658,807 / 51,487 | 131,248 / 2,199 | 527 |

15 merge では順序探索だけで source を 97.7%, DEFLATE を 95.7% 削減しました. これは state-elimination 順序の cost search が実際に支配的な差を作ることを示します.

minimization 自体は効いており, 100 merge では `101 / 3,355` から `32 / 998`, 500 merge では `501 / 95,657` から `188 / 34,652` へ縮みました. それでも 1,000 merge の未最小 DFA は `1,001 states / 376,870 groups` であり, 500〜1,000 merge の観測係数を 50,000 merges へ二次外挿すると約 9.4 億 groups です.

regex size は stepwise かつ強く superlinear なので full-size の信頼できる一点予測はできません. 参考として 0〜15 merge の増分を意図的に過小な線形で延長すると, 固定順は約 18.9 GB source / 170 MB raw-DEFLATE, beam 3 は約 431 MB / 5.86 MB です. 後者でも現在の full r50k Python artifact 211,297 bytes の約 28 倍であり, しかも約 9.4 億 transition groups の構築問題を含みません. residual quotient と順序探索は大きく効きますが, full r50k には automaton の dense graph 表現と state-elimination の式展開を止める graph/DAG lowering がなお必要です.

同じ計測と prefix ごとの runtime oracle 比較は次で再実行できます.

```bash
uv run python tools/measure_canonical_r50k.py
```

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

Makefile は 4 encoding の artifact を再生成してから, 両 examples でbinary 展開・regex compile・byte captures・Unicode / 境界ケースを検証します. Node.js では全 merge frontier pattern を V8 上で compile し, prefix-free 性・capture rank の欠落・重複・予約 rank 混入・pattern 幅を検査した上で, `tiktoken` から復元した全マージ親ペアを実際の prefix dispatch へ通します. 最後に決定的に生成した 1,008 入力について, `tiktoken`・Python API・Node.js API の token ID が一致することを検証します.

`run` target には `ARGS` で任意の引数列を渡せます. 同じ引数が 4 encoding・両言語へ渡ります.

```bash
make -C examples run ARGS='こんにちは, 世界\!'
make -C examples run ARGS='--verify'
```

## Browser demo

[examples/web](examples/web) は外部ライブラリを使わず, ブラウザ標準の `DecompressionStream` / `RegExp` / `TextEncoder` だけで ECMAScript artifact を読み込むデモです.
artifact の取得・展開, RegExp compile, tokenize, benchmark は Web Worker 内で直列実行します.

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
