import { describe, it } from "vitest";
import { gemv } from "../src/kernels/gemv";
import { matmul } from "../src/kernels/matmul";
import { matmulRef } from "../src/reference/matmul";
import { expectClose, getKiln, randomData } from "./helpers";

// [M, N, K]: K both a multiple of 4 (vec4 path) and not (scalar path), N not
// a multiple of the 4 rows per workgroup, and real decoder shapes.
const SHAPES: [number, number, number][] = [
  [1, 1, 4],
  [1, 7, 13],
  [1, 384, 384],
  [1, 1536, 384],
  [1, 51865, 384], // Whisper's vocab projection - the biggest decoder matmul
  [3, 130, 257],
  [8, 384, 1536],
];

describe.each([true, false])("gemv (subgroups=%s)", (subgroups) => {
  it.each(SHAPES)("M=%i N=%i K=%i", async (M, N, K) => {
    const kiln = await getKiln();
    if (subgroups && !kiln.features.subgroups) return;
    const x = randomData(M * K, M + K);
    const w = randomData(N * K, N);
    const bias = randomData(N, 9);
    const got = await gemv(kiln, kiln.tensor(x, [M, K]), kiln.tensor(w, [N, K]), {
      bias: kiln.tensor(bias, [N]),
      subgroups,
    }).readF32();
    expectClose(got, matmulRef(x, w, M, N, K, { transposeB: true, bias }), 1e-5, 1e-6 * Math.sqrt(K) * 4);
  });
});

describe("matmul routing", () => {
  it("sends a single-row Linear to gemv and gets the same answer", async () => {
    const kiln = await getKiln();
    const [M, N, K] = [1, 300, 384];
    const x = randomData(M * K, 1);
    const w = randomData(N * K, 2);
    const auto = await matmul(kiln, kiln.tensor(x, [M, K]), kiln.tensor(w, [N, K]), { transposeB: true }).readF32();
    expectClose(auto, matmulRef(x, w, M, N, K, { transposeB: true }), 1e-5, 1e-4);
  });
});
