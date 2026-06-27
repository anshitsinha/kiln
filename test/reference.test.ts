import { describe, expect, it } from "vitest";
import { erf, gelu, layerNormRef, softmaxRef } from "../src/reference/ops";

// The references are the spec, so they get checked against known values too.
describe("reference ops", () => {
  it("erf matches tabulated values", () => {
    // values from Abramowitz & Stegun table 7.1 / high-precision sources
    const table: [number, number][] = [
      [0, 0],
      [0.1, 0.1124629160182849],
      [0.5, 0.5204998778130465],
      [1, 0.8427007929497149],
      [2, 0.9953222650189527],
      [3, 0.9999779095030014],
      [4.5, 0.9999999998033839],
    ];
    for (const [x, want] of table) {
      expect(erf(x)).toBeCloseTo(want, 14);
      expect(erf(-x)).toBeCloseTo(-want, 14);
    }
  });

  it("gelu has the expected shape", () => {
    expect(gelu(0)).toBe(0);
    expect(gelu(1)).toBeCloseTo(0.8413447460685429, 12);
    expect(gelu(-1)).toBeCloseTo(-0.15865525393145707, 12);
    expect(gelu(10)).toBeCloseTo(10, 12);
  });

  it("softmax rows sum to 1, causal rows ignore the future", () => {
    const out = softmaxRef([1, 2, 3, 4, 5, 6], 2, 3, true);
    // row 0 of 2 with 3 cols: may see cols 0..1
    expect(out[2]).toBe(0);
    expect(out[0] + out[1]).toBeCloseTo(1, 6);
    expect(out[3] + out[4] + out[5]).toBeCloseTo(1, 6);
  });

  it("layernorm normalises each row", () => {
    const out = layerNormRef([1, 2, 3, 4], 1, 4, [1, 1, 1, 1], [0, 0, 0, 0], 0);
    const mean = out.reduce((a, b) => a + b) / 4;
    const variance = out.reduce((a, b) => a + (b - mean) ** 2, 0) / 4;
    expect(mean).toBeCloseTo(0, 6);
    expect(variance).toBeCloseTo(1, 5);
  });
});
