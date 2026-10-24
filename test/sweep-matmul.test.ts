import { it } from "vitest";
import { matmul } from "../src/kernels/matmul";
import { checkTile, type TileConfig } from "../src/kernels/matmul-tuned";
import { matmulRef } from "../src/reference/matmul";
import { expectClose, getKiln, randomData } from "./helpers";

const SHAPES: [number, number, number][] = [
  [1500, 1536, 384],
  [1500, 384, 1536],
  [1500, 1152, 384],
  [1500, 384, 384],
];

const CONFIGS: TileConfig[] = [];
for (const [wx, wy] of [[16, 16], [16, 8], [8, 16], [32, 8]])
    for (const [tm, tn] of [[4, 4], [8, 4], [4, 8], [8, 8]])
      for (const bk of [8, 16]) {
        const c = { wx, wy, tm, tn, bk };
        if (!checkTile(c)) CONFIGS.push(c);
      }

// A tuning tool, not a test: VITE_KILN_TOOLS=1 npx vitest run test/sweep-matmul.test.ts
it.runIf(import.meta.env.VITE_KILN_TOOLS)("sweep tuned matmul tiles", { timeout: 600_000 }, async () => {
  const kiln = await getKiln();
  const data = SHAPES.map(([M, N, K]) => ({
    M, N, K,
    a: kiln.tensor(randomData(M * K, 1, 1), [M, K]),
    w: kiln.tensor(randomData(N * K, 2, 1), [N, K]),
  }));
  const time = async (fn: () => void) => {
    const ts: number[] = [];
    for (let r = 0; r < 5; r++) {
      const t = await kiln.profile(async () => fn());
      ts.push(t.reduce((s, k) => s + k.ms, 0));
    }
    return ts.sort((x, y) => x - y)[2];
  };
  const row = async (name: string, opts: object) => {
    let total = 0;
    const per: string[] = [];
    for (const d of data) {
      const out = kiln.empty([d.M, d.N]);
      const run = () => void matmul(kiln, d.a, d.w, { transposeB: true, out, ...opts });
      run();
      await kiln.flush();
      const ms = await time(run);
      total += ms;
      per.push(`${((2 * d.M * d.N * d.K) / ms / 1e6).toFixed(0)}`);
      out.destroy();
    }
    console.log(`${total.toFixed(3)} ms  [${per.join(" / ")} GFLOP/s]  ${name}`);
    return total;
  };
  await row("blocked", { variant: "blocked" });
  const results: [number, string][] = [];
  for (const c of CONFIGS) {
    // correctness first, on a ragged shape
    const [M, N, K] = [130, 200, 136];
    const a = randomData(M * K, 3, 1);
    const w = randomData(N * K, 4, 1);
    const got = await matmul(kiln, kiln.tensor(a, [M, K]), kiln.tensor(w, [N, K]), { transposeB: true, variant: "tuned", tile: c }).readF32();
    expectClose(got, matmulRef(a, w, M, N, K, { transposeB: true }), 1e-5, 1e-4);
    results.push([await row(JSON.stringify(c), { variant: "tuned", tile: c }), JSON.stringify(c)]);
  }
  results.sort((x, y) => x[0] - y[0]);
  console.log("BEST:", results.slice(0, 5).map(([t, n]) => `${t.toFixed(3)} ${n}`).join("\n      "));
});
