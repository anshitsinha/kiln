import { describe, it } from "vitest";
import { gemv } from "../src/kernels/gemv";
import { gemvInt4, uploadInt4 } from "../src/kernels/gemv-int4";
import { matmul, type MatmulVariant } from "../src/kernels/matmul";
import { dequantizeInt4, quantizeInt4 } from "../src/quant/int4";
import { matmulRef } from "../src/reference/matmul";
import { gelu, mapRef } from "../src/reference/ops";
import { expectClose, getKiln, randomData } from "./helpers";

// A fused GELU must give the same numbers as matmul-then-GELU. Inputs are
// scaled so the pre-activations land around [-3, 3], where GELU actually
// bends (negative values get squashed, positive ones pass through).

describe.each(["naive", "tiled", "blocked"] as MatmulVariant[])("matmul + gelu epilogue (%s)", (variant) => {
  it.each([
    [1, 7, 5],
    [70, 130, 96],
    [128, 384, 384],
  ])("M=%i N=%i K=%i", async (M, N, K) => {
    const kiln = await getKiln();
    const a = randomData(M * K, 1, 1);
    const b = randomData(N * K, 2, 2 / Math.sqrt(K));
    const bias = randomData(N, 3, 0.5);
    const got = await matmul(kiln, kiln.tensor(a, [M, K]), kiln.tensor(b, [N, K]), {
      transposeB: true,
      bias: kiln.tensor(bias, [N]),
      variant,
      activation: "gelu",
    }).readF32();
    const want = mapRef(matmulRef(a, b, M, N, K, { transposeB: true, bias }), gelu);
    expectClose(got, want, 1e-5, 1e-5);
  });
});

describe.each([true, false])("gemv + gelu epilogue (subgroups=%s)", (subgroups) => {
  it.each([
    [1, 1536, 384],
    [3, 101, 64],
  ])("M=%i N=%i K=%i", async (M, N, K) => {
    const kiln = await getKiln();
    if (subgroups && !kiln.features.subgroups) return;
    const x = randomData(M * K, 4, 1);
    const w = randomData(N * K, 5, 2 / Math.sqrt(K));
    const bias = randomData(N, 6, 0.5);
    const got = await gemv(kiln, kiln.tensor(x, [M, K]), kiln.tensor(w, [N, K]), {
      bias: kiln.tensor(bias, [N]),
      subgroups,
      activation: "gelu",
    }).readF32();
    expectClose(got, mapRef(matmulRef(x, w, M, N, K, { transposeB: true, bias }), gelu), 1e-5, 1e-5);
  });

  it.each([
    [1, 1536, 384],
    [2, 384, 1536],
  ])("int4: M=%i N=%i K=%i", async (M, N, K) => {
    const kiln = await getKiln();
    if (subgroups && !kiln.features.subgroups) return;
    const x = randomData(M * K, 7, 1);
    const q = quantizeInt4(randomData(N * K, 8, 2 / Math.sqrt(K)), N, K);
    const bias = randomData(N, 9, 0.5);
    const got = await gemvInt4(kiln, kiln.tensor(x, [M, K]), uploadInt4(kiln, q), {
      bias: kiln.tensor(bias, [N]),
      subgroups,
      activation: "gelu",
    }).readF32();
    const want = mapRef(matmulRef(x, dequantizeInt4(q), M, N, K, { transposeB: true, bias }), gelu);
    expectClose(got, want, 1e-5, 1e-5);
  });
});

describe("matmul routes the epilogue through to gemv", () => {
  it("M=1 with transposeB keeps the activation", async () => {
    const kiln = await getKiln();
    const [N, K] = [200, 128];
    const x = randomData(K, 10, 1);
    const w = randomData(N * K, 11, 2 / Math.sqrt(K));
    const got = await matmul(kiln, kiln.tensor(x, [1, K]), kiln.tensor(w, [N, K]), { transposeB: true, activation: "gelu" }).readF32();
    expectClose(got, mapRef(matmulRef(x, w, 1, N, K, { transposeB: true }), gelu), 1e-5, 1e-5);
  });
});
