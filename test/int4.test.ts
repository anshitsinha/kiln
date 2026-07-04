import { describe, expect, it } from "vitest";
import { dequantizeInt4, GROUP, int4Bytes, quantizeInt4 } from "../src/quant/int4";
import { randomData } from "./helpers";

describe("int4 quantization (CPU)", () => {
  it("round-trips values that are exactly representable", () => {
    // one group: scale = 7/7 = 1, so integers -7..7 survive exactly
    const w = Array.from({ length: GROUP }, (_, i) => (i % 15) - 7);
    const back = dequantizeInt4(quantizeInt4(w, 1, GROUP));
    expect(Array.from(back)).toEqual(w);
  });

  it("keeps error within half a quantization step per group", () => {
    const [N, K] = [64, 384];
    const w = randomData(N * K, 3, 0.1);
    const q = quantizeInt4(w, N, K);
    const back = dequantizeInt4(q);
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < K; k++) {
        const s = q.scales[n * (K / GROUP) + Math.floor(k / GROUP)];
        expect(Math.abs(back[n * K + k] - w[n * K + k])).toBeLessThanOrEqual(s / 2 + 1e-7);
      }
    }
  });

  it("handles all-zero groups", () => {
    const back = dequantizeInt4(quantizeInt4(new Float32Array(GROUP * 2), 2, GROUP));
    expect(back.every((v) => v === 0)).toBe(true);
  });

  it("packs low nibble first", () => {
    const w = new Float32Array(GROUP);
    w[0] = 7; // max -> q = 7 -> nibble 15
    w[1] = -7; // q = -7 -> nibble 1
    const { packed } = quantizeInt4(w, 1, GROUP);
    expect(packed[0] & 0xf).toBe(15);
    expect((packed[0] >>> 4) & 0xf).toBe(1);
  });

  it("uses 0.625 bytes per weight", () => {
    expect(int4Bytes(1536, 384) / (1536 * 384)).toBe(0.625);
  });

  it("rejects K that isn't a multiple of the group size", () => {
    expect(() => quantizeInt4(new Float32Array(40), 1, 40)).toThrow(/multiple/);
  });
});
