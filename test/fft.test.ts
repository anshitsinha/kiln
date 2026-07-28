import { describe, expect, it } from "vitest";
import { dft, rfft } from "../src/audio/fft";
import { randomData } from "./helpers";

describe("mixed-radix FFT", () => {
  // 400 is the one that matters (Whisper's n_fft); the rest exercise each
  // radix and a prime length
  it.each([1, 2, 4, 5, 8, 12, 25, 30, 97, 400, 512])("matches a direct DFT at n=%i", (n) => {
    const x = randomData(n, n, 1);
    const fast = rfft(x);
    const slow = dft(x);
    for (let k = 0; k < n; k++) {
      expect(fast.re[k]).toBeCloseTo(slow.re[k], 9);
      expect(fast.im[k]).toBeCloseTo(slow.im[k], 9);
    }
  });

  it("puts a pure tone in the right bin", () => {
    const n = 400;
    const x = Float64Array.from({ length: n }, (_, j) => Math.cos((2 * Math.PI * 25 * j) / n));
    const { re } = rfft(x);
    expect(re[25]).toBeCloseTo(n / 2, 9);
    expect(Math.abs(re[24])).toBeLessThan(1e-9);
  });
});
