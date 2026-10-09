import { Kiln } from "../src";
import { gemv } from "../src/kernels/gemv";
import { defaultLanes, gemvInt4, uploadInt4, type Lanes } from "../src/kernels/gemv-int4";
import { int4Bytes, quantizeInt4 } from "../src/quant/int4";

const CASES = [
  { N: 384, K: 384, what: "attention projection" },
  { N: 1536, K: 384, what: "MLP up" },
  { N: 384, K: 1536, what: "MLP down" },
  { N: 51865, K: 384, what: "vocab projection" },
];
const RUNS = 9;

export interface DecodeResult {
  shape: string;
  what: string;
  format: string;
  ms: number;
  mb: number;
  gbps: number;
  speedup: number;
  relErr: number | null;
}

function rand(n: number, seed: number, scale: number): Float32Array {
  let s = seed;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    out[i] = ((s >>> 0) / 4294967296 - 0.5) * 2 * scale;
  }
  return out;
}

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

export async function runDecodeBench(onStatus: (s: string) => void = () => {}): Promise<DecodeResult[]> {
  const kiln = await Kiln.create();
  if (!kiln.features.timestamps) throw new Error("needs timestamp-query");
  const results: DecodeResult[] = [];

  for (const c of CASES) {
    onStatus(`${c.N}x${c.K}…`);
    const wData = rand(c.N * c.K, c.N + c.K, 0.1);
    const x = kiln.tensor(rand(c.K, 7, 1), [1, c.K]);
    const wF32 = kiln.tensor(wData, [c.N, c.K]);
    const wQ4 = uploadInt4(kiln, quantizeInt4(wData, c.N, c.K));
    const outF32 = kiln.empty([1, c.N]);
    const outQ4 = kiln.empty([1, c.N]);

    const time = async (run: () => void) => {
      run();
      await kiln.flush();
      const t = await kiln.profile(() => {
        for (let i = 0; i < RUNS; i++) run();
      });
      return median(t.map((x) => x.ms));
    };
    const msF32 = await time(() => gemv(kiln, x, wF32, { out: outF32 }));
    // sweep threads-per-row to see where the int4 kernel stops being
    // overhead-bound; the default is whatever defaultLanes(K) picks
    const lanesMs = new Map<Lanes, number>();
    for (const lanes of [4, 8, 16, 32, 64] as Lanes[]) {
      lanesMs.set(lanes, await time(() => gemvInt4(kiln, x, wQ4, { out: outQ4, lanes })));
    }
    const def = defaultLanes(c.N, c.K);
    gemvInt4(kiln, x, wQ4, { out: outQ4 }); // leave the default's output for the error check
    await kiln.flush();

    // how much the int4 output differs from the f32 output, relative to the
    // f32 output's overall magnitude (RMS), worst element
    const a = await outF32.readF32();
    const b = await outQ4.readF32();
    let rms = 0;
    for (const v of a) rms += v * v;
    rms = Math.sqrt(rms / a.length);
    let worst = 0;
    for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]) / rms);

    const bytesF32 = c.N * c.K * 4;
    const bytesQ4 = int4Bytes(c.N, c.K);
    const shape = `1×${c.N}×${c.K}`;
    results.push({ shape, what: c.what, format: "f32", ms: msF32, mb: bytesF32 / 1e6, gbps: bytesF32 / (msF32 * 1e6), speedup: 1, relErr: null });
    for (const [lanes, ms] of lanesMs) {
      results.push({
        shape,
        what: c.what,
        format: `int4, ${lanes} lanes${lanes === def ? " (default)" : ""}`,
        ms,
        mb: bytesQ4 / 1e6,
        gbps: bytesQ4 / (ms * 1e6),
        speedup: msF32 / ms,
        relErr: lanes === def ? worst : null,
      });
    }

    for (const t of [x, wF32, wQ4.packed, wQ4.params, outF32, outQ4]) t.destroy();
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
      for (const r of await runDecodeBench((s) => (status.textContent = s))) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${r.format === "f32" ? r.shape : ""}</td><td>${r.format === "f32" ? r.what : ""}</td>
          <td>${r.format}</td><td class="num">${r.ms.toFixed(4)}</td><td class="num">${fmt.format(r.mb)}</td>
          <td class="num">${fmt.format(r.gbps)}</td>
          <td class="num ${r.speedup > 1 ? "best" : ""}">${fmt.format(r.speedup)}×</td>
          <td class="num">${r.relErr === null ? "—" : r.relErr.toExponential(1)}</td>`;
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
    runDecodeBench: typeof runDecodeBench;
  }
}
window.runDecodeBench = runDecodeBench;
