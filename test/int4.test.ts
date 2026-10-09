import { describe, expect, it } from "vitest";
import { dequantizeInt4, GROUP, groupParams, int4Bytes, type Int4Mode, quantizeInt4 } from "../src/quant/int4";
import { randomData } from "./helpers";

const rms = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / a.length);
};

describe.each(["asymmetric", "symmetric"] as Int4Mode[])("int4 quantization, %s (CPU)", (mode) => {
  it("round-trips values that are exactly representable", () => {
    // symmetric: scale 1, integers -7..7; asymmetric: min -7, max 7 ->
    // scale 14/15, so use the grid it actually has
    const w =
      mode === "symmetric"
        ? Array.from({ length: GROUP }, (_, i) => (i % 15) - 7)
        : Array.from({ length: GROUP }, (_, i) => -8 + (i % 16));
    const back = dequantizeInt4(quantizeInt4(w, 1, GROUP, mode));
    expect(Array.from(back)).toEqual(w);
  });

  it("keeps error within half a quantization step per group", () => {
    const [N, K] = [64, 384];
    const w = randomData(N * K, 3, 0.1);
    const q = quantizeInt4(w, N, K, mode);
    const back = dequantizeInt4(q);
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < K; k++) {
        const [s] = groupParams(q.params, n * (K / GROUP) + Math.floor(k / GROUP));
        // + a little for the f16-rounded offset
        expect(Math.abs(back[n * K + k] - w[n * K + k])).toBeLessThanOrEqual(s / 2 + 1e-4 * 0.1);
      }
    }
  });

  it("handles all-zero and constant groups", () => {
    const back = dequantizeInt4(quantizeInt4(new Float32Array(GROUP * 2), 2, GROUP, mode));
    expect(back.every((v) => v === 0)).toBe(true);
    if (mode === "asymmetric") {
      const c = dequantizeInt4(quantizeInt4(new Float32Array(GROUP).fill(0.25), 1, GROUP, mode));
      expect(c.every((v) => v === 0.25)).toBe(true);
    }
  });

  it("packs low nibble first", () => {
    const w = new Float32Array(GROUP);
    w[0] = 7; // the max -> nibble 15 either way
    w[1] = -7; // symmetric: q = -7 -> nibble 1; asymmetric: the min -> 0
    const { packed } = quantizeInt4(w, 1, GROUP, mode);
    expect(packed[0] & 0xf).toBe(15);
    expect((packed[0] >>> 4) & 0xf).toBe(mode === "symmetric" ? 1 : 0);
  });
});

describe("int4 quantization", () => {
  it("asymmetric beats symmetric on skewed groups, and isn't worse on centred ones", () => {
    const [N, K] = [32, 256];
    const centred = randomData(N * K, 4, 0.1);
    const skewed = centred.map((v) => Math.abs(v) + 0.02);
    for (const [w, factor] of [
      [skewed, 0.6],
      [centred, 1.0],
    ] as const) {
      const sym = rms(dequantizeInt4(quantizeInt4(w, N, K, "symmetric")), w);
      const asym = rms(dequantizeInt4(quantizeInt4(w, N, K, "asymmetric")), w);
      expect(asym).toBeLessThan(sym * factor);
    }
  });

  it("uses 0.625 bytes per weight (the offset rides along for free)", () => {
    expect(int4Bytes(1536, 384) / (1536 * 384)).toBe(0.625);
  });

  it("rejects K that isn't a multiple of the group size", () => {
    expect(() => quantizeInt4(new Float32Array(40), 1, 40)).toThrow(/multiple/);
  });
});
