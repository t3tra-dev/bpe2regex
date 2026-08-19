const MAGIC = [0x42, 0x32, 0x52, 0x58];
const FORMAT_VERSION = 1;
const ECMASCRIPT_COMPATIBILITY = 1;
const RANK_ALPHABET =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const RANK_RADIX = RANK_ALPHABET.length;
const ENCODINGS = new Map([
  [0, { slug: "r50k", name: "r50k_base", reservedRanks: [] }],
  [1, { slug: "o200k", name: "o200k_base", reservedRanks: [] }],
  [2, { slug: "p50k", name: "p50k_base", reservedRanks: [50256] }],
  [3, { slug: "cl100k", name: "cl100k_base", reservedRanks: [] }],
]);

let encodingSelect;
let input;
let status;
let statusText;
let tokenList;
let tokenCount;
let metrics;
let artifactRoot;

if (typeof document !== "undefined") {
  encodingSelect = document.querySelector("#encoding");
  input = document.querySelector("#input");
  status = document.querySelector("#status");
  statusText = document.querySelector("#status-text");
  tokenList = document.querySelector("#tokens");
  tokenCount = document.querySelector("#token-count");
  metrics = {
    binarySize: document.querySelector("#binary-size"),
    containerSize: document.querySelector("#container-size"),
    regexpSize: document.querySelector("#regexp-size"),
    loadTime: document.querySelector("#load-time"),
    firstTime: document.querySelector("#first-time"),
    averageTime: document.querySelector("#average-time"),
    throughput: document.querySelector("#throughput"),
    benchmarkRuns: document.querySelector("#benchmark-runs"),
  };
  const configuredArtifactRoot = new URL(
    document.location.href,
  ).searchParams.get("artifacts");
  artifactRoot = new URL(
    configuredArtifactRoot ?? "./.artifacts/",
    document.baseURI,
  );
}

const byteFormatter = new Intl.NumberFormat("ja-JP", {
  maximumFractionDigits: 2,
});
const countFormatter = new Intl.NumberFormat("ja-JP");
const tokenDecoder = new TextDecoder("utf-8", { fatal: true });
const textEncoder = new TextEncoder();
let demoWorker;
let workerReady = false;
let activeEncoding = "";
let activeGeneration = 0;
let nextRequestId = 0;
let latestTokenizeId = 0;

function rankCodeWidth(tokenCount_) {
  if (!Number.isSafeInteger(tokenCount_) || tokenCount_ <= 0) {
    throw new Error("token count must be a positive safe integer");
  }
  let width = 1;
  let capacity = RANK_RADIX;
  while (capacity < tokenCount_) {
    width += 1;
    capacity *= RANK_RADIX;
  }
  return width;
}

function encodeRank(rank, width) {
  if (
    !Number.isSafeInteger(width) ||
    width <= 0 ||
    !Number.isSafeInteger(rank) ||
    rank < 0 ||
    rank >= RANK_RADIX ** width
  ) {
    throw new Error(`rank ${rank} does not fit in ${width} base62 digits`);
  }
  const encoded = Array(width).fill(RANK_ALPHABET[0]);
  let remainder = rank;
  for (let position = width - 1; position >= 0; position -= 1) {
    const digit = remainder % RANK_RADIX;
    remainder = Math.floor(remainder / RANK_RADIX);
    encoded[position] = RANK_ALPHABET[digit];
  }
  return encoded.join("");
}

class BinaryReader {
  constructor(value) {
    this.value = value;
    this.position = 0;
    this.decoder = new TextDecoder("utf-8", { fatal: true });
  }

  read(size) {
    const end = this.position + size;
    if (!Number.isSafeInteger(size) || size < 0 || end > this.value.length) {
      throw new Error("truncated regex artifact");
    }
    const result = this.value.subarray(this.position, end);
    this.position = end;
    return result;
  }

  uint() {
    let value = 0;
    let shift = 0;
    while (shift <= 63) {
      const byte = this.read(1)[0];
      value += (byte & 0x7f) * 2 ** shift;
      if (byte < 0x80) {
        if (!Number.isSafeInteger(value)) throw new Error("oversized integer");
        return value;
      }
      shift += 7;
    }
    throw new Error("oversized integer in regex artifact");
  }

  text() {
    return this.decoder.decode(this.read(this.uint()));
  }

  texts() {
    return Array.from({ length: this.uint() }, () => this.text());
  }

  ranks(tokenCount_) {
    const count = this.uint();
    let width = 1;
    for (let maximum = tokenCount_ - 1; maximum >= 256; maximum /= 256) {
      width += 1;
    }
    const content = this.read(count * width);
    const ranks = [];
    for (let offset = 0; offset < content.length; offset += width) {
      let rank = 0;
      for (let byte = 0; byte < width; byte += 1) {
        rank += content[offset + byte] * 256 ** byte;
      }
      if (!Number.isSafeInteger(rank) || rank >= tokenCount_) {
        throw new Error(`capture rank is outside the vocabulary: ${rank}`);
      }
      ranks.push(rank);
    }
    return ranks;
  }

  rankTables(tokenCount_) {
    return Array.from({ length: this.uint() }, () => this.ranks(tokenCount_));
  }

  finish() {
    if (this.position !== this.value.length) {
      throw new Error("trailing data in regex artifact");
    }
  }
}

function compareCandidates(left, right) {
  return left[0] - right[0] || left[1] - right[1];
}

class MinHeap {
  constructor() {
    this.values = [];
  }

  get length() {
    return this.values.length;
  }

  push(value) {
    const values = this.values;
    let index = values.length;
    values.push(value);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (compareCandidates(values[parent], value) <= 0) break;
      values[index] = values[parent];
      index = parent;
    }
    values[index] = value;
  }

  pop() {
    const values = this.values;
    if (values.length === 0) return undefined;
    const result = values[0];
    const tail = values.pop();
    if (values.length === 0) return result;
    let index = 0;
    while (true) {
      const left = index * 2 + 1;
      if (left >= values.length) break;
      const right = left + 1;
      let child = left;
      if (
        right < values.length &&
        compareCandidates(values[right], values[left]) < 0
      ) {
        child = right;
      }
      if (compareCandidates(values[child], tail) >= 0) break;
      values[index] = values[child];
      index = child;
    }
    values[index] = tail;
    return result;
  }
}

function exactPattern(source) {
  return new RegExp(`^(?:${source})$`);
}

function resolveRank(patterns, value) {
  let rank = 0;
  for (let bit = 0; bit < patterns.length; bit += 1) {
    if (patterns[bit].test(value)) rank += 2 ** bit;
  }
  return rank;
}

function wellFormed(text) {
  if (typeof text.toWellFormed === "function") return text.toWellFormed();
  let result = "";
  for (let index = 0; index < text.length; index += 1) {
    const current = text.charCodeAt(index);
    if (current >= 0xd800 && current <= 0xdbff) {
      const next = text.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        result += text[index] + text[index + 1];
        index += 1;
      } else {
        result += "\ufffd";
      }
    } else if (current >= 0xdc00 && current <= 0xdfff) {
      result += "\ufffd";
    } else {
      result += text[index];
    }
  }
  return result;
}

async function inflateRaw(content) {
  if (typeof DecompressionStream !== "function") {
    throw new Error("このブラウザは DecompressionStream に対応していません");
  }
  let decompressor;
  try {
    decompressor = new DecompressionStream("deflate-raw");
  } catch (error) {
    throw new Error("このブラウザは raw DEFLATE 展開に対応していません", {
      cause: error,
    });
  }
  const stream = new Blob([content]).stream().pipeThrough(decompressor);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function decodeArtifact(value) {
  const reader = new BinaryReader(value);
  const magic = reader.read(MAGIC.length);
  if (!MAGIC.every((byte, index) => magic[index] === byte)) {
    throw new Error("invalid regex artifact magic");
  }
  const version = reader.read(1)[0];
  const encodingId = reader.read(1)[0];
  const compatibility = reader.read(1)[0];
  if (version !== FORMAT_VERSION) {
    throw new Error(`unsupported regex artifact version: ${version}`);
  }
  if (!ENCODINGS.has(encodingId)) {
    throw new Error(`unsupported encoding identifier: ${encodingId}`);
  }
  if (compatibility !== ECMASCRIPT_COMPATIBILITY) {
    throw new Error(`unexpected compatibility identifier: ${compatibility}`);
  }
  const tokenCount_ = reader.uint();
  const baseTokenCount = reader.uint();
  const rankWidth = reader.uint();
  const byteSources = reader.texts();
  const mergePrefixes = reader.texts();
  const mergeSources = reader.texts();
  const mergeRanks = reader.rankTables(tokenCount_);
  const pretokenizerSource = reader.text();
  reader.finish();
  if (
    tokenCount_ <= baseTokenCount ||
    baseTokenCount !== 256 ||
    rankWidth !== rankCodeWidth(tokenCount_) ||
    byteSources.length !== Math.ceil(Math.log2(baseTokenCount)) ||
    mergeSources.length !== mergePrefixes.length ||
    mergeRanks.length !== mergePrefixes.length
  ) {
    throw new Error("invalid regex artifact dimensions");
  }
  return {
    ...ENCODINGS.get(encodingId),
    tokenCount: tokenCount_,
    baseTokenCount,
    rankWidth,
    byteSources,
    mergePrefixes,
    mergeSources,
    mergeRanks,
    pretokenizerSource,
  };
}

class RegexBPE {
  constructor(data) {
    this.encoding = data.name;
    this.tokenCount = data.tokenCount;
    this.baseTokenCount = data.baseTokenCount;
    this.rankWidth = data.rankWidth;
    this.reservedRanks = new Set(data.reservedRanks);
    this.bytePatterns = data.byteSources.map(exactPattern);
    this.mergePrefixMap = new Map(
      data.mergePrefixes.map((prefix, index) => [prefix, index]),
    );
    this.mergeSources = data.mergeSources;
    this.mergeRanks = data.mergeRanks;
    this.mergePatterns = new Map();
    this.mergeCache = new Map();
    this.pretokenizer = new RegExp(data.pretokenizerSource, "uy");

    const observed = new Set();
    for (let byte = 0; byte < 256; byte += 1) {
      observed.add(resolveRank(this.bytePatterns, String.fromCharCode(byte)));
    }
    if (observed.size !== this.baseTokenCount) {
      throw new Error("base-rank bit regexes are not bijective");
    }
  }

  mergeRank(left, right) {
    const pair =
      encodeRank(left, this.rankWidth) + encodeRank(right, this.rankWidth);
    const cached = this.mergeCache.get(pair);
    if (cached !== undefined) return cached;

    let patternIndex;
    let prefixLength = 0;
    for (; prefixLength <= pair.length; prefixLength += 1) {
      const candidate = this.mergePrefixMap.get(pair.slice(0, prefixLength));
      if (candidate !== undefined) {
        patternIndex = candidate;
        break;
      }
    }
    if (patternIndex === undefined) return undefined;

    let pattern = this.mergePatterns.get(patternIndex);
    if (pattern === undefined) {
      pattern = exactPattern(this.mergeSources[patternIndex]);
      this.mergePatterns.set(patternIndex, pattern);
    }
    const match = pattern.exec(pair.slice(prefixLength));
    if (match === null) return undefined;

    let captureIndex = -1;
    for (let index = 1; index < match.length; index += 1) {
      if (match[index] !== undefined) {
        if (captureIndex >= 0) {
          throw new Error("merge pattern selected multiple captures");
        }
        captureIndex = index - 1;
      }
    }
    const rank = this.mergeRanks[patternIndex][captureIndex];
    if (
      rank === undefined ||
      rank < this.baseTokenCount ||
      rank >= this.tokenCount ||
      this.reservedRanks.has(rank)
    ) {
      throw new Error("merge pattern matched without a valid child rank");
    }
    this.mergeCache.set(pair, rank);
    return rank;
  }

  fullmatch(input_) {
    if (!(input_ instanceof Uint8Array)) {
      throw new TypeError("fullmatch() requires a Uint8Array");
    }
    const source = new Uint8Array(input_);
    if (source.length === 0) {
      return { tokenIds: [], captures: [] };
    }

    const tokenIds = Array.from(source, (byte) =>
      resolveRank(this.bytePatterns, String.fromCharCode(byte)),
    );
    const count = tokenIds.length;
    const starts = Array.from({ length: count }, (_, index) => index);
    const ends = Array.from({ length: count }, (_, index) => index + 1);
    const previous = new Int32Array(count);
    const following = new Int32Array(count);
    const versions = new Uint32Array(count);
    for (let index = 0; index < count; index += 1) {
      previous[index] = index - 1;
      following[index] = index + 1 < count ? index + 1 : -1;
    }

    const candidates = new MinHeap();
    const pushCandidate = (leftIndex) => {
      const rightIndex = following[leftIndex];
      if (rightIndex < 0) return;
      const rank = this.mergeRank(tokenIds[leftIndex], tokenIds[rightIndex]);
      if (rank !== undefined) {
        candidates.push([
          rank,
          leftIndex,
          versions[leftIndex],
          rightIndex,
          versions[rightIndex],
        ]);
      }
    };
    for (let index = 0; index + 1 < count; index += 1) pushCandidate(index);

    while (candidates.length) {
      const candidate = candidates.pop();
      const [rank, left, leftVersion, right, rightVersion] = candidate;
      if (
        versions[left] !== leftVersion ||
        versions[right] !== rightVersion ||
        following[left] !== right ||
        previous[right] !== left
      ) {
        continue;
      }

      tokenIds[left] = rank;
      ends[left] = ends[right];
      const next = following[right];
      following[left] = next;
      if (next >= 0) previous[next] = left;
      following[right] = -2;
      previous[right] = -2;
      versions[left] += 1;
      versions[right] += 1;
      if (previous[left] >= 0) pushCandidate(previous[left]);
      pushCandidate(left);
    }

    const finalIds = [];
    const captures = [];
    for (let index = 0; index >= 0; index = following[index]) {
      finalIds.push(tokenIds[index]);
      captures.push(source.slice(starts[index], ends[index]));
    }
    return { tokenIds: finalIds, captures };
  }

  split(text) {
    if (typeof text !== "string")
      throw new TypeError("split() requires a string");
    const source = wellFormed(text);
    const pieces = [];
    let position = 0;
    while (position < source.length) {
      this.pretokenizer.lastIndex = position;
      const match = this.pretokenizer.exec(source);
      if (match === null || match.index !== position || match[0].length === 0) {
        throw new Error(
          `pre-tokenizer left a gap at UTF-16 offset ${position}`,
        );
      }
      pieces.push(match[0]);
      position = this.pretokenizer.lastIndex;
    }
    return pieces;
  }

  tokenizeOrdinary(text) {
    const tokenIds = [];
    const captures = [];
    for (const piece of this.split(text)) {
      const match = this.fullmatch(textEncoder.encode(piece));
      tokenIds.push(...match.tokenIds);
      captures.push(...match.captures);
    }
    return { tokenIds, captures };
  }
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return "—";
  const units = ["B", "KiB", "MiB", "GiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit + 1 < units.length) {
    amount /= 1024;
    unit += 1;
  }
  return `${byteFormatter.format(amount)} ${units[unit]}`;
}

function formatDuration(milliseconds) {
  if (!Number.isFinite(milliseconds)) return "—";
  const digits = milliseconds < 1 ? 3 : milliseconds < 10 ? 2 : 1;
  return `${milliseconds.toFixed(digits)} ms`;
}

function displayToken(value) {
  try {
    return tokenDecoder.decode(value);
  } catch {
    return `${Array.from(
      value,
      (byte) => `\\x${byte.toString(16).padStart(2, "0")}`,
    ).join("")}`;
  }
}

function setStatus(message, state_) {
  statusText.textContent = message;
  status.dataset.state = state_;
}

function resetTokenMetrics() {
  metrics.firstTime.textContent = "—";
  metrics.averageTime.textContent = "—";
  metrics.throughput.textContent = "—";
  metrics.benchmarkRuns.textContent = "—";
}

function renderTokens(result) {
  tokenList.replaceChildren();
  if (result.tokenIds.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "空の入力です";
    tokenList.append(empty);
  } else {
    const fragment = document.createDocumentFragment();
    for (let index = 0; index < result.tokenIds.length; index += 1) {
      const item = document.createElement("li");
      item.className = "token";
      const rank = document.createElement("span");
      rank.className = "token-id";
      rank.textContent = String(result.tokenIds[index]);
      const text = document.createElement("span");
      text.className = "token-text";
      text.textContent = displayToken(result.captures[index]);
      item.append(rank, text);
      fragment.append(item);
    }
    tokenList.append(fragment);
  }
  tokenCount.textContent = `${countFormatter.format(result.tokenIds.length)} tokens`;
}

function resetArtifactMetrics() {
  metrics.binarySize.textContent = "—";
  metrics.containerSize.textContent = "—";
  metrics.regexpSize.textContent = "—";
  metrics.loadTime.textContent = "—";
}

function requestTokenize() {
  if (!workerReady) return;
  const id = ++nextRequestId;
  latestTokenizeId = id;
  setStatus(`${activeEncoding} · tokenize 中…`, "loading");
  demoWorker.postMessage({
    type: "tokenize",
    id,
    generation: activeGeneration,
    text: input.value,
  });
}

function loadEncoding(slug) {
  const generation = ++activeGeneration;
  const id = ++nextRequestId;
  workerReady = false;
  latestTokenizeId = 0;
  encodingSelect.disabled = true;
  resetArtifactMetrics();
  resetTokenMetrics();
  setStatus(`${slug} artifact を取得しています…`, "loading");
  demoWorker.postMessage({
    type: "load",
    id,
    generation,
    slug,
    artifactRoot: artifactRoot.href,
  });
}

function handleWorkerMessage(event) {
  const message = event.data;
  if (message.generation !== activeGeneration) return;

  if (message.type === "loaded") {
    workerReady = true;
    activeEncoding = message.encoding;
    encodingSelect.disabled = false;
    metrics.binarySize.textContent = formatBytes(message.binarySize);
    metrics.containerSize.textContent = formatBytes(message.containerSize);
    metrics.regexpSize.textContent = formatBytes(message.regexpSize);
    metrics.loadTime.textContent = formatDuration(message.loadTime);
    setStatus(`${message.encoding} artifact を読み込みました`, "ready");
    requestTokenize();
  } else if (message.type === "tokenized") {
    if (message.id !== latestTokenizeId) return;
    metrics.firstTime.textContent = formatDuration(message.firstTime);
    metrics.averageTime.textContent = "計測中…";
    metrics.throughput.textContent = "—";
    metrics.benchmarkRuns.textContent = "—";
    renderTokens(message);
    setStatus(
      `${message.encoding} · ${countFormatter.format(message.encodedSize)} UTF-8 bytes`,
      "ready",
    );
  } else if (message.type === "benchmarked") {
    if (message.id !== latestTokenizeId) return;
    metrics.averageTime.textContent = formatDuration(message.averageTime);
    metrics.throughput.textContent =
      message.bytesPerSecond === 0
        ? "—"
        : `${formatBytes(message.bytesPerSecond)}/s`;
    metrics.benchmarkRuns.textContent = countFormatter.format(message.runs);
  } else if (message.type === "error") {
    console.error(message.message);
    if (message.operation === "load") {
      workerReady = false;
      encodingSelect.disabled = false;
    } else if (message.id !== latestTokenizeId) {
      return;
    }
    setStatus(message.message, "error");
    tokenList.replaceChildren();
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = "処理を完了できませんでした";
    tokenList.append(item);
    tokenCount.textContent = "—";
  }
}

if (typeof document !== "undefined") {
  demoWorker = new Worker(new URL("./worker.js", import.meta.url), {
    type: "module",
  });
  demoWorker.addEventListener("message", handleWorkerMessage);
  demoWorker.addEventListener("error", (event) => {
    workerReady = false;
    encodingSelect.disabled = false;
    setStatus(`Worker error: ${event.message}`, "error");
  });

  input.addEventListener("input", requestTokenize);

  encodingSelect.addEventListener("change", () => {
    loadEncoding(encodingSelect.value);
  });

  loadEncoding(encodingSelect.value);
}

export { decodeArtifact, RegexBPE, inflateRaw, wellFormed };
