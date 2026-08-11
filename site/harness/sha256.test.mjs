// Conformance: site/sha256.js vs node:crypto across lengths and chunk
// boundaries (the block-straddling cases are where streaming digests break).
// If this fails, the model verifier is untrustworthy.
import { createHash, randomBytes } from "node:crypto";
import { Sha256 } from "../sha256.js";

const LENGTHS = [0, 1, 3, 55, 56, 57, 63, 64, 65, 127, 128, 1000, 65536, 1_000_000];
const SPLITS = [1, 7, 63, 64, 65, 4096, 65535];

let failures = 0;
for (const len of LENGTHS) {
  const data = randomBytes(len);
  const want = createHash("sha256").update(data).digest("hex");
  // one-shot
  if (new Sha256().update(new Uint8Array(data)).hex() !== want) {
    console.error(`FAIL one-shot len=${len}`);
    failures += 1;
  }
  // every split size
  for (const split of SPLITS) {
    const h = new Sha256();
    for (let off = 0; off < len; off += split) {
      h.update(new Uint8Array(data.subarray(off, Math.min(off + split, len))));
    }
    if (h.hex() !== want) {
      console.error(`FAIL len=${len} split=${split}`);
      failures += 1;
    }
  }
}
// pinned vector: sha256("abc")
const abc = new Sha256().update(new TextEncoder().encode("abc")).hex();
if (abc !== "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
  console.error("FAIL FIPS vector sha256('abc')");
  failures += 1;
}
if (failures) {
  console.error(`sha256 conformance: ${failures} failure(s)`);
  process.exit(1);
}
console.log(`sha256 conformance: OK (${LENGTHS.length} lengths x ${SPLITS.length} splits + FIPS vector)`);
