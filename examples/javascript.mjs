import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { inflateRawSync } from "node:zlib";

const MAGIC = Buffer.from("B2RX", "ascii");
const FORMAT_VERSION = 1;
const RANK_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const RANK_RADIX = RANK_ALPHABET.length;
const ENCODINGS = new Map([
  [0, { name: "r50k_base", reservedRanks: [] }],
  [1, { name: "o200k_base", reservedRanks: [] }],
  [2, { name: "p50k_base", reservedRanks: [50256] }],
  [3, { name: "cl100k_base", reservedRanks: [] }],
]);
const ECMASCRIPT_COMPATIBILITY = 1;

function rankCodeWidth(tokenCount) {
  if (!Number.isSafeInteger(tokenCount) || tokenCount <= 0) {
    throw new Error("token count must be a positive safe integer");
  }
  let width = 1;
  let capacity = RANK_RADIX;
  while (capacity < tokenCount) {
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
  constructor(path) {
    this.value = inflateRawSync(readFileSync(path));
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

  ranks(tokenCount) {
    const count = this.uint();
    let width = 1;
    for (let maximum = tokenCount - 1; maximum >= 256; maximum /= 256) {
      width += 1;
    }
    const content = this.read(count * width);
    const ranks = [];
    for (let offset = 0; offset < content.length; offset += width) {
      let rank = 0;
      for (let byte = 0; byte < width; byte += 1) {
        rank += content[offset + byte] * 256 ** byte;
      }
      if (!Number.isSafeInteger(rank) || rank >= tokenCount) {
        throw new Error(`capture rank is outside the vocabulary: ${rank}`);
      }
      ranks.push(rank);
    }
    return ranks;
  }

  rankTables(tokenCount) {
    return Array.from({ length: this.uint() }, () => this.ranks(tokenCount));
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

function decodeArtifact(path) {
  const reader = new BinaryReader(path);
  if (!reader.read(MAGIC.length).equals(MAGIC)) {
    throw new Error("invalid regex artifact magic");
  }
  const version = reader.read(1)[0];
  const encoding = reader.read(1)[0];
  const compatibility = reader.read(1)[0];
  if (version !== FORMAT_VERSION) {
    throw new Error(`unsupported regex artifact version: ${version}`);
  }
  if (!ENCODINGS.has(encoding)) {
    throw new Error(`unsupported encoding identifier: ${encoding}`);
  }
  if (compatibility !== ECMASCRIPT_COMPATIBILITY) {
    throw new Error(`unexpected compatibility identifier: ${compatibility}`);
  }
  const tokenCount = reader.uint();
  const baseTokenCount = reader.uint();
  const rankWidth = reader.uint();
  const byteSources = reader.texts();
  const mergeBuckets = reader.texts();
  const mergeBucketRanks = reader.rankTables(tokenCount);
  const pretokenizerSource = reader.text();
  reader.finish();
  if (
    tokenCount <= baseTokenCount ||
    baseTokenCount !== 256 ||
    rankWidth !== rankCodeWidth(tokenCount) ||
    byteSources.length !== Math.ceil(Math.log2(baseTokenCount)) ||
    mergeBuckets.length === 0 ||
    mergeBucketRanks.length !== mergeBuckets.length
  ) {
    throw new Error("invalid regex artifact dimensions");
  }
  const encodingData = ENCODINGS.get(encoding);
  return {
    encoding: encodingData.name,
    reservedRanks: encodingData.reservedRanks,
    tokenCount,
    baseTokenCount,
    rankWidth,
    byteSources,
    mergeBuckets,
    mergeBucketRanks,
    pretokenizerSource,
  };
}

export class ECMAScriptRegexBPE {
  constructor(data) {
    this.encoding = data.encoding;
    this.tokenCount = data.tokenCount;
    this.baseTokenCount = data.baseTokenCount;
    this.bytePatterns = data.byteSources.map(exactPattern);
    this.mergeBucketSources = data.mergeBuckets;
    this.mergeBucketRanks = data.mergeBucketRanks;
    this.mergeBucketCount = data.mergeBuckets.length;
    this.mergeBucketPatterns = new Map();
    this.pretokenizer = new RegExp(data.pretokenizerSource, "uy");
    this.mergeCache = new Map();
    this.rankWidth = data.rankWidth;
    this.reservedRanks = new Set(data.reservedRanks);
  }

  mergeRank(left, right) {
    const key = left * this.tokenCount + right;
    const cached = this.mergeCache.get(key);
    if (cached !== undefined) return cached;
    const bucket = key % this.mergeBucketCount;
    let pattern = this.mergeBucketPatterns.get(bucket);
    if (pattern === undefined) {
      pattern = exactPattern(this.mergeBucketSources[bucket]);
      this.mergeBucketPatterns.set(bucket, pattern);
    }
    const pair =
      encodeRank(left, this.rankWidth) + encodeRank(right, this.rankWidth);
    const match = pattern.exec(pair);
    if (match === null) return undefined;
    let rank;
    let captureIndex = -1;
    for (let index = 1; index < match.length; index += 1) {
      if (match[index] !== undefined) {
        if (captureIndex >= 0) {
          throw new Error("merge bucket selected multiple captures");
        }
        captureIndex = index - 1;
      }
    }
    if (captureIndex >= 0) {
      rank = this.mergeBucketRanks[bucket][captureIndex];
    }
    if (
      rank === undefined ||
      rank < this.baseTokenCount ||
      this.reservedRanks.has(rank)
    ) {
      throw new Error("merge bucket matched without a child-rank capture");
    }
    if (rank >= this.tokenCount) throw new Error(`invalid merge rank: ${rank}`);
    this.mergeCache.set(key, rank);
    return rank;
  }

  validateStructure() {
    const baseRanks = new Set();
    for (let byte = 0; byte < 256; byte += 1) {
      baseRanks.add(
        resolveRank(this.bytePatterns, String.fromCharCode(byte)),
      );
    }
    if (
      baseRanks.size !== this.baseTokenCount ||
      !Array.from(
        { length: this.baseTokenCount },
        (_, rank) => baseRanks.has(rank),
      ).every(Boolean)
    ) {
      throw new Error("base-rank bit regexes are not bijective");
    }

    const mergeRanks = new Set();
    let maxBucketRules = 0;
    for (let bucket = 0; bucket < this.mergeBucketCount; bucket += 1) {
      const source = this.mergeBucketSources[bucket];
      const pattern = exactPattern(source);
      this.mergeBucketPatterns.set(bucket, pattern);
      let bucketRules = 0;
      const captureRanks = this.mergeBucketRanks[bucket];
      const captureCount = Array.from(source.matchAll(/\(\)/g)).length;
      if (source.includes("(?<m") || captureCount !== captureRanks.length) {
        throw new Error("merge bucket capture table width differs");
      }
      for (const rank of captureRanks) {
        if (
          !Number.isSafeInteger(rank) ||
          rank < this.baseTokenCount ||
          rank >= this.tokenCount ||
          this.reservedRanks.has(rank) ||
          mergeRanks.has(rank)
        ) {
          throw new Error(`invalid or duplicate merge capture rank: ${rank}`);
        }
        mergeRanks.add(rank);
        bucketRules += 1;
      }
      maxBucketRules = Math.max(maxBucketRules, bucketRules);
    }

    const expectedMergeRules =
      this.tokenCount - this.baseTokenCount - this.reservedRanks.size;
    if (mergeRanks.size !== expectedMergeRules) {
      throw new Error(
        `merge capture count differs: ${mergeRanks.size} != ${expectedMergeRules}`,
      );
    }
    for (let rank = this.baseTokenCount; rank < this.tokenCount; rank += 1) {
      if (!this.reservedRanks.has(rank) && !mergeRanks.has(rank)) {
        throw new Error(`missing merge capture rank: ${rank}`);
      }
    }
    if (maxBucketRules > this.mergeBucketCount) {
      throw new Error(
        `merge bucket is wider than the split bound: ${maxBucketRules}`,
      );
    }
    return { mergeBucketCount: this.mergeBucketCount, maxBucketRules };
  }

  fullmatch(input) {
    if (!(input instanceof Uint8Array)) {
      throw new TypeError("fullmatch() requires a Uint8Array or Buffer");
    }
    const source = new Uint8Array(input);
    if (source.length === 0) {
      return { tokenIds: [], spans: [], captures: [], pathCount: 1 };
    }

    const tokenIds = Array.from(source, (byte) =>
      resolveRank(this.bytePatterns, String.fromCharCode(byte)),
    );
    const tokenCount = tokenIds.length;
    const starts = Array.from({ length: tokenCount }, (_, index) => index);
    const ends = Array.from({ length: tokenCount }, (_, index) => index + 1);
    const previous = new Int32Array(tokenCount);
    const following = new Int32Array(tokenCount);
    const versions = new Uint32Array(tokenCount);
    for (let index = 0; index < tokenCount; index += 1) {
      previous[index] = index - 1;
      following[index] = index + 1 < tokenCount ? index + 1 : -1;
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
    for (let index = 0; index + 1 < tokenCount; index += 1) {
      pushCandidate(index);
    }

    while (candidates.length) {
      const [rank, leftIndex, leftVersion, rightIndex, rightVersion] =
        candidates.pop();
      if (
        versions[leftIndex] !== leftVersion ||
        versions[rightIndex] !== rightVersion ||
        following[leftIndex] !== rightIndex ||
        previous[rightIndex] !== leftIndex
      ) {
        continue;
      }

      tokenIds[leftIndex] = rank;
      ends[leftIndex] = ends[rightIndex];
      const nextIndex = following[rightIndex];
      following[leftIndex] = nextIndex;
      if (nextIndex >= 0) previous[nextIndex] = leftIndex;
      following[rightIndex] = -2;
      previous[rightIndex] = -2;
      versions[leftIndex] += 1;
      versions[rightIndex] += 1;

      const previousIndex = previous[leftIndex];
      if (previousIndex >= 0) pushCandidate(previousIndex);
      pushCandidate(leftIndex);
    }

    const finalIds = [];
    const spans = [];
    for (let index = 0; index >= 0; index = following[index]) {
      finalIds.push(tokenIds[index]);
      spans.push([starts[index], ends[index]]);
    }
    const captures = spans.map(([start, end]) => source.slice(start, end));
    return { tokenIds: finalIds, spans, captures, pathCount: 1 };
  }

  split(text) {
    if (typeof text !== "string") throw new TypeError("split() requires a string");
    const source = wellFormed(text);
    const pieces = [];
    let position = 0;
    while (position < source.length) {
      this.pretokenizer.lastIndex = position;
      const match = this.pretokenizer.exec(source);
      if (match === null || match.index !== position || match[0].length === 0) {
        throw new Error(`pre-tokenizer left a gap at UTF-16 offset ${position}`);
      }
      pieces.push(match[0]);
      position = this.pretokenizer.lastIndex;
    }
    return pieces;
  }

  encodeOrdinary(text) {
    const encoder = new TextEncoder();
    const tokenIds = [];
    for (const piece of this.split(text)) {
      tokenIds.push(...this.fullmatch(encoder.encode(piece)).tokenIds);
    }
    return tokenIds;
  }

  tokenizeOrdinary(text) {
    const encoder = new TextEncoder();
    const tokens = [];
    for (const piece of this.split(text)) {
      tokens.push(...this.fullmatch(encoder.encode(piece)).captures);
    }
    return tokens;
  }
}

export function loadECMAScriptProgram(path) {
  return new ECMAScriptRegexBPE(decodeArtifact(path));
}

function equalIds(actual, expected) {
  return (
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}

const tokenDecoder = new TextDecoder("utf-8", { fatal: true });

function displayToken(token) {
  let text;
  try {
    text = tokenDecoder.decode(token);
  } catch {
    const escapedBytes = Array.from(
      token,
      (value) => `\\x${value.toString(16).padStart(2, "0")}`,
    ).join("");
    return `"${escapedBytes}"`;
  }

  let escaped = "";
  for (const character of text) {
    const codepoint = character.codePointAt(0);
    if (character === "\\") escaped += "\\\\";
    else if (character === '"') escaped += '\\"';
    else if (character === "\n") escaped += "\\n";
    else if (character === "\r") escaped += "\\r";
    else if (character === "\t") escaped += "\\t";
    else if (
      codepoint < 0x20 ||
      (codepoint >= 0x7f && codepoint <= 0x9f) ||
      codepoint === 0x2028 ||
      codepoint === 0x2029
    ) {
      escaped += `\\u${codepoint.toString(16).padStart(4, "0")}`;
    } else escaped += character;
  }
  return `"${escaped}"`;
}

function formatTokens(tokens) {
  return `[${tokens.map(displayToken).join(", ")}]`;
}

function verify(program) {
  const r50kCases = [
    ["", []],
    ["hello world", [31373, 995]],
    [
      " 日本語とEnglish 12345!\n",
      [10545, 245, 98, 17312, 105, 45739, 252, 30201, 15823, 17031, 2231, 0, 198],
    ],
    ["\u001ca", [216, 64]],
    ["a\n\n", [64, 628]],
    ["<|endoftext|>", [27, 91, 437, 1659, 5239, 91, 29]],
    ["a\ud800b", [64, 4210, 65]],
  ];
  const o200kCases = [
    ["", []],
    ["hello world", [24912, 2375]],
    [
      " 日本語とEnglish 12345!\n",
      [17428, 40909, 5330, 28881, 220, 7633, 2548, 4175],
    ],
    ["\u001ca", [216, 64]],
    ["a\n\n", [64, 279]],
    ["<|endoftext|>", [27, 91, 419, 1440, 919, 91, 29]],
    ["a\ud800b", [64, 3251, 65]],
  ];
  const p50kCases = [
    ["", []],
    ["hello world", [31373, 995]],
    [
      " 日本語とEnglish 12345!\n",
      [10545, 245, 98, 17312, 105, 45739, 252, 30201, 15823, 17031, 2231, 0, 198],
    ],
    ["\u001ca", [216, 64]],
    ["a\n\n", [64, 628]],
    ["<|endoftext|>", [27, 91, 437, 1659, 5239, 91, 29]],
    ["a\ud800b", [64, 4210, 65]],
    ["hello    world", [31373, 50258, 995]],
  ];
  const cl100kCases = [
    ["", []],
    ["hello world", [15339, 1917]],
    [
      " 日本語とEnglish 12345!\n",
      [76502, 22656, 45918, 252, 19732, 23392, 220, 4513, 1774, 4999],
    ],
    ["\u001ca", [216, 64]],
    ["a\n\n", [64, 271]],
    ["<|endoftext|>", [27, 91, 8862, 728, 428, 91, 29]],
    ["a\ud800b", [64, 5809, 65]],
    ["!hello1234567", [0, 15339, 4513, 10961, 22]],
  ];
  const casesByEncoding = new Map([
    ["r50k_base", r50kCases],
    ["p50k_base", p50kCases],
    ["cl100k_base", cl100kCases],
    ["o200k_base", o200kCases],
  ]);
  const cases = casesByEncoding.get(program.encoding);
  if (cases === undefined) throw new Error(`no cases for ${program.encoding}`);
  const structure = program.validateStructure();
  for (const [caseIndex, [text, expected]] of cases.entries()) {
    const actual = program.encodeOrdinary(text);
    if (!equalIds(actual, expected)) {
      throw new Error(`token IDs differ for text case ${caseIndex}: ${actual}`);
    }
    const reconstructed = Buffer.concat(
      program.tokenizeOrdinary(text).map((value) => Buffer.from(value)),
    );
    const expectedBytes = Buffer.from(new TextEncoder().encode(wellFormed(text)));
    if (!reconstructed.equals(expectedBytes)) {
      throw new Error(`token bytes differ for text case ${caseIndex}`);
    }
  }
  const bytes = Buffer.from("hello world");
  const match = program.fullmatch(bytes);
  const reconstructed = Buffer.concat(
    match.captures.map((value) => Buffer.from(value)),
  );
  const expectedHello = cases.find(([text]) => text === "hello world")[1];
  if (!reconstructed.equals(bytes) || !equalIds(match.tokenIds, expectedHello)) {
    throw new Error("byte captures do not reconstruct hello world");
  }
  console.info(
    `javascript ${program.encoding}: ok (${cases.length} text cases, ` +
      `${structure.mergeBucketCount} merge buckets, ` +
      `max ${structure.maxBucketRules} rules/bucket)`,
  );
}

export function main(arguments_ = process.argv.slice(2)) {
  let artifact = new URL("../.artifacts/r50k/ecmascript.bin", import.meta.url);
  let verifyRequested = false;
  const textParts = [];
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index];
    if (argument === "--artifact") {
      index += 1;
      if (index >= arguments_.length) {
        throw new Error("--artifact requires a path");
      }
      artifact = arguments_[index];
    } else if (argument === "--verify") {
      verifyRequested = true;
    } else {
      textParts.push(argument);
    }
  }

  const program = loadECMAScriptProgram(artifact);
  if (verifyRequested) {
    verify(program);
  } else {
    const text = textParts.length ? textParts.join(" ") : "hello world";
    process.stdout.write(`${formatTokens(program.tokenizeOrdinary(text))}\n`);
  }
}

if (
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  main();
}
