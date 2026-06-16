import { describe, expect, it } from "vitest";
import { matmul, type MatmulVariant } from "../src/kernels/matmul";
import { matmulRef } from "../src/reference/matmul";
import { expectClose, getKiln, randomData } from "./helpers";

const VARIANTS: MatmulVariant[] = ["naive", "tiled", "blocked"];

// Shapes chosen to poke at edges: smaller than one tile, not a multiple of
// any tile size, a single row (decoder step), long K, and real Whisper-tiny
// layer shapes.
const SHAPES: [number, number, number][] = [
  [1, 1, 1],
  [3, 5, 7],
  [17, 33, 65],
  [64, 64, 64],
  [1, 384, 384], // one decoder token through a projection
  [100, 130, 300],
  [1500, 384, 64], // encoder: attention head projection-ish
  [70, 1536, 384], // MLP up-projection, few tokens
];

describe.each(VARIANTS)("matmul (%s)", (variant) => {
  it.each(SHAPES)("M=%i N=%i K=%i", async (M, N, K) => {
    const kiln = await getKiln();
    const a = randomData(M * K, M * 31 + K);
    const b = randomData(K * N, N * 17 + K);
    const got = await matmul(kiln, kiln.tensor(a, [M, K]), kiln.tensor(b, [K, N]), { variant }).readF32();
    // f32 sums of K terms of magnitude <=1: error grows roughly with sqrt(K)
    expectClose(got, matmulRef(a, b, M, N, K), 1e-5, 1e-6 * Math.sqrt(K) * 4);
  });

  it("supports transposed B (Linear weights) and fused bias", async () => {
    const kiln = await getKiln();
    const [M, N, K] = [37, 70, 129];
    const a = randomData(M * K, 1);
    const w = randomData(N * K, 2); // [N, K]
    const bias = randomData(N, 3);
    const got = await matmul(kiln, kiln.tensor(a, [M, K]), kiln.tensor(w, [N, K]), {
      variant,
      transposeB: true,
      bias: kiln.tensor(bias, [N]),
    }).readF32();
    expectClose(got, matmulRef(a, w, M, N, K, { transposeB: true, bias }), 1e-5, 1e-4);
  });

  it("batches, with B either shared or per-batch", async () => {
    const kiln = await getKiln();
    const [batch, M, N, K] = [3, 20, 24, 40];
    const a = randomData(batch * M * K, 4);
    const shared = randomData(K * N, 5);
    const perBatch = randomData(batch * K * N, 6);

    const gotShared = await matmul(kiln, kiln.tensor(a, [batch, M, K]), kiln.tensor(shared, [K, N]), { variant }).readF32();
    const gotPer = await matmul(kiln, kiln.tensor(a, [batch, M, K]), kiln.tensor(perBatch, [batch, K, N]), { variant }).readF32();
    for (let i = 0; i < batch; i++) {
      const ai = a.subarray(i * M * K, (i + 1) * M * K);
      expectClose(gotShared.subarray(i * M * N, (i + 1) * M * N), matmulRef(ai, shared, M, N, K), 1e-5, 1e-4);
      const bi = perBatch.subarray(i * K * N, (i + 1) * K * N);
      expectClose(gotPer.subarray(i * M * N, (i + 1) * M * N), matmulRef(ai, bi, M, N, K), 1e-5, 1e-4);
    }
  });
});

describe("matmul shape checks", () => {
  it("rejects mismatched inner dimensions", async () => {
    const kiln = await getKiln();
    expect(() => matmul(kiln, kiln.empty([4, 5]), kiln.empty([6, 7]))).toThrow(/inner dims/);
  });
});
