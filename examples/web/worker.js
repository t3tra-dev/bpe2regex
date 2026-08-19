import { decodeArtifact, RegexBPE, inflateRaw, wellFormed } from "./app.js";

const textEncoder = new TextEncoder();
let program = null;
let activeGeneration = 0;

async function loadArtifact(request) {
  const loadStart = performance.now();
  const artifactUrl = new URL(
    `${request.slug}/ecmascript.bin`,
    request.artifactRoot,
  );
  const response = await fetch(artifactUrl);
  if (!response.ok) {
    throw new Error(`artifact を取得できませんでした: HTTP ${response.status}`);
  }

  const compressed = await response.arrayBuffer();
  const inflated = await inflateRaw(compressed);
  const data = decodeArtifact(inflated);
  if (data.slug !== request.slug) {
    throw new Error(`artifact encoding が一致しません: ${data.name}`);
  }

  const nextProgram = new RegexBPE(data);
  const regexpBytes = new Blob([
    ...data.byteSources,
    ...data.mergeSources,
    data.pretokenizerSource,
  ]).size;
  program = nextProgram;
  activeGeneration = request.generation;
  self.postMessage({
    type: "loaded",
    id: request.id,
    generation: request.generation,
    encoding: data.name,
    binarySize: compressed.byteLength,
    containerSize: inflated.byteLength,
    regexpSize: regexpBytes,
    loadTime: performance.now() - loadStart,
  });
}

function tokenize(request) {
  if (program === null || request.generation !== activeGeneration) {
    throw new Error("tokenizer artifact が読み込まれていません");
  }

  const encodedSize = textEncoder.encode(wellFormed(request.text)).byteLength;
  const firstStart = performance.now();
  const result = program.tokenizeOrdinary(request.text);
  const firstTime = performance.now() - firstStart;
  self.postMessage(
    {
      type: "tokenized",
      id: request.id,
      generation: request.generation,
      encoding: program.encoding,
      encodedSize,
      firstTime,
      tokenIds: result.tokenIds,
      captures: result.captures,
    },
    result.captures.map((capture) => capture.buffer),
  );

  const benchmarkStart = performance.now();
  let runs = 0;
  let elapsed = 0;
  do {
    program.tokenizeOrdinary(request.text);
    runs += 1;
    elapsed = performance.now() - benchmarkStart;
  } while (elapsed < 40 && runs < 500);

  const averageTime = elapsed / runs;
  const bytesPerSecond =
    encodedSize === 0 || elapsed === 0
      ? 0
      : (encodedSize * runs * 1000) / elapsed;
  self.postMessage({
    type: "benchmarked",
    id: request.id,
    generation: request.generation,
    averageTime,
    bytesPerSecond,
    runs,
  });
}

async function handle(request) {
  if (
    request === null ||
    typeof request !== "object" ||
    !Number.isSafeInteger(request.id) ||
    !Number.isSafeInteger(request.generation)
  ) {
    throw new TypeError("invalid worker request");
  }
  if (request.type === "load") {
    await loadArtifact(request);
  } else if (request.type === "tokenize") {
    tokenize(request);
  } else {
    throw new Error(`unknown worker request: ${request.type}`);
  }
}

let operation = Promise.resolve();
self.addEventListener("message", (event) => {
  const request = event.data;
  operation = operation.then(async () => {
    try {
      await handle(request);
    } catch (error) {
      self.postMessage({
        type: "error",
        operation: request?.type,
        id: request?.id,
        generation: request?.generation,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  });
});
