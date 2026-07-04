import { describe, it } from "vitest";
import { gemvInt4, uploadInt4 } from "../src/kernels/gemv-int4";
import { dequantizeInt4, quantizeInt4 } from "../src/quant/int4";
import { matmulRef } from "../src/reference/matmul";
import { expectClose, getKiln, randomData } from "./helpers";

// The kernel's job is to compute X @ dequant(W)^T exactly as the CPU would -
// quantization error is a property of the format (tested in int4.test.ts),
// not of the kernel, so the reference uses the dequantized weights.
const SHAPES: [number, number, number][] = [
  [1, 1, 32],
  [1, 7, 64],
  [1, 384, 384],
  [1, 1536, 384],
  [1, 384, 1536],
  [1, 51865, 384], // vocab projection
  [4, 130, 256],
  [8, 384, 384],
];

describe.each([true, false])("gemvInt4 (subgroups=%s)", (subgroups) => {
  it.each(SHAPES)("M=%i N=%i K=%i", async (M, N, K) => {
    const kiln = await getKiln();
    if (subgroups && !kiln.features.subgroups) return;
    const x = randomData(M * K, M + K, 1);
    const q = quantizeInt4(randomData(N * K, N + 1, 0.2), N, K);
    const bias = randomData(N, 5, 0.5);
    const got = await gemvInt4(kiln, kiln.tensor(x, [M, K]), uploadInt4(kiln, q), {
      bias: kiln.tensor(bias, [N]),
      subgroups,
    }).readF32();
    expectClose(got, matmulRef(x, dequantizeInt4(q), M, N, K, { transposeB: true, bias }), 1e-5, 1e-6 * Math.sqrt(K) * 4);
  });
});
