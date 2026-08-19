import { createInterface } from "node:readline";

import { loadECMAScriptProgram } from "../examples/javascript.mjs";

if (process.argv.length !== 3) {
  throw new Error("usage: node tests/javascript_validate_merges.mjs ARTIFACT");
}

const program = loadECMAScriptProgram(process.argv[2]);
program.validateStructure();
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
let count = 0;
for await (const line of lines) {
  const [left, right, expected] = line.split(",").map(Number);
  const actual = program.mergeRank(left, right);
  if (actual !== expected) {
    throw new Error(
      `merge rank differs for (${left}, ${right}): ${actual} != ${expected}`,
    );
  }
  count += 1;
}
process.stdout.write(`${count}\n`);
