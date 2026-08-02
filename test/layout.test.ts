import { describe, expect, it } from "vitest";
import { convOutLength, embed, im2col, mergeHeads, splitHeads, type Conv1dShape } from "../src/kernels/layout";
import { matmul } from "../src/kernels/matmul";
import { expectClose, getKiln, randomData } from "./helpers";

/** Direct 1-D convolution, the definition. x is [C, T], w is [Cout, C, K]. */
function conv1dRef(x: Float32Array, w: Float32Array, C: number, T: number, Cout: number, K: number, stride: number, pad: number) {
  const tOut = Math.floor((T + 2 * pad - K) / stride) + 1;
  const out = new Float32Array(tOut * Cout); // [tOut, Cout]
  for (let t = 0; t < tOut; t++)
    for (let o = 0; o < Cout; o++) {
      let s = 0;
      for (let c = 0; c < C; c++)
        for (let j = 0; j < K; j++) {
          const src = t * stride + j - pad;
          if (src >= 0 && src < T) s += x[c * T + src] * w[(o * C + c) * K + j];
        }
      out[t * Cout + o] = s;
    }
  return out;
}

describe("im2col + matmul = conv1d", () => {
  // Whisper's two convs (at reduced length so the reference is quick), plus
  // a [T, C]-layout input like the second conv gets
  it.each([
    ["conv1-like", { C: 80, T: 100, kernel: 3, stride: 1, pad: 1, layout: "CT" }, 64],
    ["conv2-like", { C: 64, T: 100, kernel: 3, stride: 2, pad: 1, layout: "CT" }, 64],
    ["odd sizes", { C: 5, T: 9, kernel: 3, stride: 2, pad: 1, layout: "CT" }, 7],
  ] as [string, Conv1dShape, number][])("%s", async (_n, s, Cout) => {
    const kiln = await getKiln();
    const x = randomData(s.C * s.T, 1, 1);
    const w = randomData(Cout * s.C * s.kernel, 2, 0.3);
    const cols = im2col(kiln, kiln.tensor(x, [s.C, s.T]), s);
    expect(cols.shape).toEqual([convOutLength(s), s.C * s.kernel]);
    const got = await matmul(kiln, cols, kiln.tensor(w, [Cout, s.C * s.kernel]), { transposeB: true }).readF32();
    expectClose(got, conv1dRef(x, w, s.C, s.T, Cout, s.kernel, s.stride, s.pad), 1e-5, 1e-5);
  });

  it("reads [T, C] activations the same as [C, T]", async () => {
    const kiln = await getKiln();
    const [C, T] = [6, 11];
    const xCT = randomData(C * T, 3);
    const xTC = new Float32Array(C * T);
    for (let c = 0; c < C; c++) for (let t = 0; t < T; t++) xTC[t * C + c] = xCT[c * T + t];
    const shape = { C, T, kernel: 3, stride: 2, pad: 1 };
    const a = await im2col(kiln, kiln.tensor(xCT, [C, T]), { ...shape, layout: "CT" }).readF32();
    const b = await im2col(kiln, kiln.tensor(xTC, [T, C]), { ...shape, layout: "TC" }).readF32();
    expect(Array.from(b)).toEqual(Array.from(a));
  });
});

describe("heads", () => {
  it("splits and merges back to the original", async () => {
    const kiln = await getKiln();
    const [T, H, D] = [7, 3, 4];
    const x = randomData(T * H * D, 4);
    const split = splitHeads(kiln, kiln.tensor(x, [T, H * D]), H);
    const s = await split.readF32();
    // spot-check the layout: [h, t, d] <- [t, h*D + d]
    expect(s[(2 * T + 5) * D + 1]).toBe(x[5 * H * D + 2 * D + 1]);
    expect(Array.from(await mergeHeads(kiln, split).readF32())).toEqual(Array.from(x));
  });

  it("appends into a KV cache at a row offset", async () => {
    const kiln = await getKiln();
    const [H, D, cap] = [2, 4, 10];
    const cache = kiln.empty([H, cap, D]);
    const a = randomData(3 * H * D, 5);
    const b = randomData(2 * H * D, 6);
    splitHeads(kiln, kiln.tensor(a, [3, H * D]), H, { out: cache, rowOffset: 0 });
    splitHeads(kiln, kiln.tensor(b, [2, H * D]), H, { out: cache, rowOffset: 3 });
    const c = await cache.readF32();
    // head 1, row 4 (2nd row of the second append), dim 2
    expect(c[(1 * cap + 4) * D + 2]).toBe(b[1 * H * D + 1 * D + 2]);
    // rows past what was written stay zero
    expect(c[(0 * cap + 7) * D]).toBe(0);
    expect(() => splitHeads(kiln, kiln.tensor(b, [2, H * D]), H, { out: cache, rowOffset: 9 })).toThrow(/can't write/);
  });
});

describe("embed", () => {
  it("adds token and position rows", async () => {
    const kiln = await getKiln();
    const [V, P, C] = [10, 8, 4];
    const tok = randomData(V * C, 7);
    const pos = randomData(P * C, 8);
    const ids = [3, 0, 9];
    const got = await embed(kiln, kiln.tensor(tok, [V, C]), kiln.tensor(pos, [P, C]), kiln.tensor(ids, [3], "u32"), 2).readF32();
    const want = new Float32Array(3 * C);
    ids.forEach((id, r) => {
      for (let c = 0; c < C; c++) want[r * C + c] = tok[id * C + c] + pos[(2 + r) * C + c];
    });
    expectClose(got, want, 0, 0);
  });
});
