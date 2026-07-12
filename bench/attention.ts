import { Kiln } from "../src";
import { attention, attentionUnfused } from "../src/kernels/attention";

const H = 6;
const D = 64;
const CASES = [
  { Tq: 1500, Tk: 1500, causal: false, what: "encoder self-attention" },
  { Tq: 100, Tk: 100, causal: true, what: "decoder self-attn, prompt" },
  { Tq: 1, Tk: 448, causal: true, what: "decoder self-attn, 1 token vs full cache" },
  { Tq: 1, Tk: 1500, causal: false, what: "cross-attn, 1 token" },
  { Tq: 4, Tk: 1500, causal: false, what: "cross-attn, 4 tokens (beam)" },
];
const RUNS = 7;

export interface AttentionResult {
  shape: string;
  what: string;
  unfusedMs: number;
  fusedMs: number;
  intermediateMB: number;
}

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

export async function runAttentionBench(onStatus: (s: string) => void = () => {}): Promise<AttentionResult[]> {
  const kiln = await Kiln.create();
  if (!kiln.features.timestamps) throw new Error("needs timestamp-query");
  const results: AttentionResult[] = [];

  for (const c of CASES) {
    onStatus(`${H}x${c.Tq}x${c.Tk}…`);
    const fill = (n: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) * 0.5);
    const q = kiln.tensor(fill(H * c.Tq * D), [H, c.Tq, D]);
    const k = kiln.tensor(fill(H * c.Tk * D), [H, c.Tk, D]);
    const v = kiln.tensor(fill(H * c.Tk * D), [H, c.Tk, D]);

    const measure = async (run: () => void, kernelsPerRun: number) => {
      run();
      await kiln.flush();
      const t = await kiln.profile(() => {
        for (let i = 0; i < RUNS; i++) run();
      });
      // sum each run's kernels, then take the median run
      const perRun: number[] = [];
      for (let r = 0; r < RUNS; r++) {
        perRun.push(t.slice(r * kernelsPerRun, (r + 1) * kernelsPerRun).reduce((a, b) => a + b.ms, 0));
      }
      return median(perRun);
    };
    const unfusedMs = await measure(() => attentionUnfused(kiln, q, k, v, { causal: c.causal }), 3);
    const fusedMs = await measure(() => attention(kiln, q, k, v, { causal: c.causal }), 1);

    results.push({
      shape: `${H}×${c.Tq}×${c.Tk}`,
      what: c.what,
      unfusedMs,
      fusedMs,
      intermediateMB: (2 * H * c.Tq * c.Tk * 4) / 1e6,
    });
    q.destroy();
    k.destroy();
    v.destroy();
    await new Promise((r) => setTimeout(r, 0));
  }
  onStatus(`done · ${kiln.adapterName}`);
  kiln.destroy();
  return results;
}

const button = document.getElementById("run") as HTMLButtonElement | null;
if (button) {
  const status = document.getElementById("status")!;
  const table = document.getElementById("results")!;
  const tbody = table.querySelector("tbody")!;
  const fmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
  button.addEventListener("click", async () => {
    button.disabled = true;
    tbody.innerHTML = "";
    try {
      for (const r of await runAttentionBench((s) => (status.textContent = s))) {
        const speedup = r.unfusedMs / r.fusedMs;
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${r.shape}</td><td>${r.what}</td><td class="num">${r.unfusedMs.toFixed(3)}</td>
          <td class="num">${r.fusedMs.toFixed(3)}</td><td class="num ${speedup > 1 ? "best" : ""}">${fmt.format(speedup)}×</td>
          <td class="num">${fmt.format(r.intermediateMB)} MB</td>`;
        tbody.appendChild(tr);
      }
      table.hidden = false;
    } catch (e) {
      status.textContent = `error: ${(e as Error).message}`;
    } finally {
      button.disabled = false;
    }
  });
}

declare global {
  interface Window {
    runAttentionBench: typeof runAttentionBench;
  }
}
window.runAttentionBench = runAttentionBench;
