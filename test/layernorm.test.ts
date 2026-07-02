import { describe, expect, it } from "vitest";
import { layerNorm } from "../src/kernels/layernorm";
import { layerNormRef } from "../src/reference/ops";
import { expectClose, getKiln, randomData } from "./helpers";

describe("layerNorm", () => {
  // d_model sizes: Whisper tiny/base/small/medium, plus odd and tiny ones
  it.each([
    [1, 1],
    [3, 7],
    [1500, 384],
    [4, 512],
    [2, 768],
    [1, 1024],
    [2, 1000],
  ])("rows=%i cols=%i", async (rows, cols) => {
    const kiln = await getKiln();
    const x = randomData(rows * cols, rows * 7 + cols, 3);
    const gamma = randomData(cols, 11, 2);
    const beta = randomData(cols, 12, 1);
    const got = await layerNorm(kiln, kiln.tensor(x, [rows, cols]), kiln.tensor(gamma, [cols]), kiln.tensor(beta, [cols])).readF32();
    expectClose(got, layerNormRef(x, rows, cols, gamma, beta), 1e-4, 1e-5);
  });

  it("stays accurate when the mean is large compared to the spread", async () => {
    // E[x^2] - E[x]^2 would lose almost every significant digit here in f32
    const kiln = await getKiln();
    const cols = 384;
    const x = randomData(cols, 5, 0.01).map((v) => v + 1000);
    const ones = new Float32Array(cols).fill(1);
    const zeros = new Float32Array(cols);
    const got = await layerNorm(kiln, kiln.tensor(x, [1, cols]), kiln.tensor(ones, [cols]), kiln.tensor(zeros, [cols])).readF32();
    // f32 inputs near 1000 only carry ~6e-5 absolute precision, so the
    // normalised values (spread ~0.006) can only be good to ~1e-2
    expectClose(got, layerNormRef(x, 1, cols, ones, zeros), 1e-2, 1e-2);
  });

  it("fuses a residual add and writes the sum out", async () => {
    const kiln = await getKiln();
    const [rows, cols] = [20, 384];
    const x = randomData(rows * cols, 21, 2);
    const r = randomData(rows * cols, 22, 2);
    const gamma = randomData(cols, 23, 2);
    const beta = randomData(cols, 24, 1);
    const sumOut = kiln.empty([rows, cols]);
    const got = await layerNorm(kiln, kiln.tensor(x, [rows, cols]), kiln.tensor(gamma, [cols]), kiln.tensor(beta, [cols]), {
      residual: kiln.tensor(r, [rows, cols]),
      sumOut,
    }).readF32();
    const sum = Float32Array.from(x, (v, i) => v + r[i]);
    expectClose(await sumOut.readF32(), sum, 0, 0);
    expectClose(got, layerNormRef(sum, rows, cols, gamma, beta), 1e-4, 1e-5);
  });

  it("checks gamma/beta sizes", async () => {
    const kiln = await getKiln();
    expect(() => layerNorm(kiln, kiln.empty([2, 8]), kiln.empty([4]), kiln.empty([8]))).toThrow(/gamma\/beta/);
  });
});
