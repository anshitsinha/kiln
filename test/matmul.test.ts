import { describe, expect, it } from "vitest";
import { DEFAULT_TILE, matmul, type MatmulVariant } from "../src/kernels/matmul";
import { checkTile, type TileConfig } from "../src/kernels/matmul-tuned";
import { matmulRef } from "../src/reference/matmul";
import { gelu, mapRef } from "../src/reference/ops";
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

describe("matmul (tuned: A @ W^T, vec4 loads)", () => {
  const TILES: TileConfig[] = [DEFAULT_TILE, { wx: 16, wy: 16, tm: 4, tn: 4, bk: 16 }, { wx: 32, wy: 8, tm: 8, tn: 4, bk: 16 }];

  it.each([
    [1, 4, 4],
    [17, 33, 68],
    [130, 200, 136], // ragged against every tile
    [300, 384, 384],
    [1500, 384, 64],
  ])("M=%i N=%i K=%i, every tile, + bias + gelu", async (M, N, K) => {
    const kiln = await getKiln();
    const a = randomData(M * K, M + 1, 1);
    const w = randomData(N * K, N + 2, 1 / Math.sqrt(K));
    const bias = randomData(N, 3, 0.5);
    const want = mapRef(matmulRef(a, w, M, N, K, { transposeB: true, bias }), gelu);
    for (const tile of TILES) {
      const got = await matmul(kiln, kiln.tensor(a, [M, K]), kiln.tensor(w, [N, K]), {
        transposeB: true,
        variant: "tuned",
        tile,
        bias: kiln.tensor(bias, [N]),
        activation: "gelu",
      }).readF32();
      expectClose(got, want, 1e-5, 1e-5);
    }
  });

  it("batches with per-batch weights", async () => {
    const kiln = await getKiln();
    const [batch, M, N, K] = [3, 20, 24, 40];
    const a = randomData(batch * M * K, 4);
    const w = randomData(batch * N * K, 5);
    const got = await matmul(kiln, kiln.tensor(a, [batch, M, K]), kiln.tensor(w, [batch, N, K]), { transposeB: true, variant: "tuned" }).readF32();
    for (let i = 0; i < batch; i++) {
      const want = matmulRef(a.subarray(i * M * K, (i + 1) * M * K), w.subarray(i * N * K, (i + 1) * N * K), M, N, K, { transposeB: true });
      expectClose(got.subarray(i * M * N, (i + 1) * M * N), want, 1e-5, 1e-4);
    }
  });

  it("is the default for A @ W^T with K % 4 == 0, and falls back otherwise", async () => {
    const kiln = await getKiln();
    const [M, N, K] = [40, 30, 22]; // K % 4 != 0 -> blocked
    const a = randomData(M * K, 6);
    const w = randomData(N * K, 7);
    const got = await matmul(kiln, kiln.tensor(a, [M, K]), kiln.tensor(w, [N, K]), { transposeB: true, variant: "tuned" }).readF32();
    expectClose(got, matmulRef(a, w, M, N, K, { transposeB: true }), 1e-5, 1e-4);
  });

  it("rejects tiles that don't fit", () => {
    expect(checkTile({ wx: 16, wy: 16, tm: 8, tn: 8, bk: 32 })).toMatch(/workgroup memory/);
    expect(checkTile({ wx: 32, wy: 16, tm: 4, tn: 4, bk: 16 })).toMatch(/threads/);
    expect(checkTile(DEFAULT_TILE)).toBeNull();
  });
});

describe("matmul shape checks", () => {
  it("rejects mismatched inner dimensions", async () => {
    const kiln = await getKiln();
    expect(() => matmul(kiln, kiln.empty([4, 5]), kiln.empty([6, 7]))).toThrow(/inner dims/);
  });
});
