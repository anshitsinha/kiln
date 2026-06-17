import { Kiln } from "../src";
import { matmul, type MatmulVariant } from "../src/kernels/matmul";

const VARIANTS: MatmulVariant[] = ["naive", "tiled", "blocked"];

// Square sizes for raw throughput, plus the shapes Whisper-tiny actually
// runs (1500 encoder frames, d_model 384, MLP 1536).
const CASES: { M: number; N: number; K: number; what: string; transposeB?: boolean }[] = [
  { M: 1024, N: 1024, K: 1024, what: "square" },
  { M: 2048, N: 2048, K: 2048, what: "square" },
  { M: 1500, N: 1536, K: 384, what: "whisper-tiny encoder MLP up", transposeB: true },
  { M: 1500, N: 384, K: 1536, what: "whisper-tiny encoder MLP down", transposeB: true },
  { M: 1500, N: 1152, K: 384, what: "whisper-tiny fused QKV", transposeB: true },
  { M: 1, N: 1536, K: 384, what: "decoder step (1 token)", transposeB: true },
];

const RUNS = 7;

export interface Result {
  shape: string;
  what: string;
  variant: MatmulVariant;
  ms: number;
  gflops: number;
}

export async function runMatmulBench(onStatus: (s: string) => void = () => {}): Promise<Result[]> {
  const kiln = await Kiln.create();
  if (!kiln.features.timestamps) throw new Error("This benchmark needs the timestamp-query feature.");
  const results: Result[] = [];

  for (const c of CASES) {
    const a = kiln.tensor(new Float32Array(c.M * c.K).fill(0.5), [c.M, c.K]);
    const b = kiln.tensor(new Float32Array(c.K * c.N).fill(0.25), c.transposeB ? [c.N, c.K] : [c.K, c.N]);
    const out = kiln.empty([c.M, c.N]);

    for (const variant of VARIANTS) {
      onStatus(`${c.M}x${c.N}x${c.K} ${variant}…`);
      const run = () => matmul(kiln, a, b, { variant, transposeB: c.transposeB, out });
      run(); // warm-up: pipeline compile
      await kiln.flush();
      const timings = await kiln.profile(() => {
        for (let i = 0; i < RUNS; i++) run();
      });
      const ms = timings.map((t) => t.ms).sort((x, y) => x - y)[Math.floor(timings.length / 2)];
      results.push({
        shape: `${c.M}×${c.N}×${c.K}`,
        what: c.what,
        variant,
        ms,
        gflops: (2 * c.M * c.N * c.K) / (ms * 1e6),
      });
      await new Promise((r) => setTimeout(r, 0));
    }
    a.destroy();
    b.destroy();
    out.destroy();
  }
  onStatus(`done · ${kiln.adapterName}`);
  kiln.destroy();
  return results;
}

// --- page --------------------------------------------------------------------

const button = document.getElementById("run") as HTMLButtonElement | null;
if (button) {
  const status = document.getElementById("status")!;
  const table = document.getElementById("results")!;
  const tbody = table.querySelector("tbody")!;
  const fmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });

  button.addEventListener("click", async () => {
    button.disabled = true;
    tbody.innerHTML = "";
    try {
      const results = await runMatmulBench((s) => (status.textContent = s));
      table.hidden = false;
      for (const r of results) {
        const naive = results.find((x) => x.shape === r.shape && x.variant === "naive")!;
        const best = Math.max(...results.filter((x) => x.shape === r.shape).map((x) => x.gflops));
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${r.variant === "naive" ? r.shape : ""}</td><td>${r.variant === "naive" ? r.what : ""}</td>
          <td>${r.variant}</td><td class="num">${r.ms.toFixed(3)}</td>
          <td class="num ${r.gflops === best ? "best" : ""}">${fmt.format(r.gflops)}</td>
          <td class="num">${fmt.format(r.gflops / naive.gflops)}×</td>`;
        tbody.appendChild(tr);
      }
    } catch (e) {
      status.textContent = `error: ${(e as Error).message}`;
    } finally {
      button.disabled = false;
    }
  });
}

declare global {
  interface Window {
    runMatmulBench: typeof runMatmulBench;
  }
}
window.runMatmulBench = runMatmulBench;
