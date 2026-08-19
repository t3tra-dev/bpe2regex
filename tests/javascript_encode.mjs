import { createInterface } from "node:readline";

import { loadECMAScriptProgram } from "../examples/javascript.mjs";

if (process.argv.length !== 3) {
  throw new Error("usage: node tests/javascript_encode.mjs ARTIFACT");
}

const program = loadECMAScriptProgram(process.argv[2]);
program.validateStructure();
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  process.stdout.write(`${JSON.stringify(program.encodeOrdinary(JSON.parse(line)))}\n`);
}
