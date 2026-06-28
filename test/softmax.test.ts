import { describe, expect, it } from "vitest";
import { softmax } from "../src/kernels/softmax";
import { softmaxRef } from "../src/reference/ops";
import { expectClose, getKiln, randomData } from "./helpers";

describe("softmax", () => {
  // [rows, cols]: tiny, one element, wider than the workgroup, and Whisper
  // attention shapes (1500 encoder positions; 448 decoder positions)
  it.each([
    [1, 1],
    [3, 5],
    [4, 300],
    [64, 1500],
    [1, 51865], // softmax over the whole vocabulary
  ])("rows=%i cols=%i", async (rows, cols) => {
    const kiln = await getKiln();
    const x = randomData(rows * cols, rows + cols, 8);
    const got = await softmax(kiln, kiln.tensor(x, [rows, cols])).readF32();
    expectClose(got, softmaxRef(x, rows, cols), 1e-5, 1e-7);
  });

  it("applies a scale before exponentiating (attention 1/sqrt(d))", async () => {
    const kiln = await getKiln();
    const [rows, cols] = [8, 64];
    const x = randomData(rows * cols, 1, 10);
    const scale = 1 / Math.sqrt(64);
    const got = await softmax(kiln, kiln.tensor(x, [rows, cols]), { scale }).readF32();
    expectClose(got, softmaxRef(x.map((v) => v * scale), rows, cols), 1e-5, 1e-7);
  });

  it("is stable for large inputs (no overflow)", async () => {
    const kiln = await getKiln();
    const x = [1000, 1001, 999, -1000];
    const got = await softmax(kiln, kiln.tensor(x, [1, 4])).readF32();
    expect(got.every(Number.isFinite)).toBe(true);
    expectClose(got, softmaxRef(x, 1, 4), 1e-5, 1e-7);
  });

  it.each([
    [6, 6], // square: plain lower triangle
    [3, 10], // fewer rows than cols: like decoding against a cache
  ])("causal mask, rows=%i cols=%i", async (rows, cols) => {
    const kiln = await getKiln();
    const x = randomData(rows * cols, 7, 4);
    const got = await softmax(kiln, kiln.tensor(x, [rows, cols]), { causal: true }).readF32();
    expectClose(got, softmaxRef(x, rows, cols, true), 1e-5, 1e-7);
  });

  it("applies the causal mask per matrix in a batch (heads)", async () => {
    const kiln = await getKiln();
    const [heads, T] = [3, 5];
    const x = randomData(heads * T * T, 9, 4);
    const got = await softmax(kiln, kiln.tensor(x, [heads, T, T]), { causal: true }).readF32();
    for (let h = 0; h < heads; h++) {
      const slice = (a: Float32Array) => a.subarray(h * T * T, (h + 1) * T * T);
      expectClose(slice(got), softmaxRef(slice(x), T, T, true), 1e-5, 1e-7);
    }
  });
});
