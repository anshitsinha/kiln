import { pipeline } from "@huggingface/transformers";
import { Kiln } from "../src";
import { decodeAudio } from "../src/audio/decode";
import { corpusWer } from "../src/eval/wer";
import { Tokenizer } from "../src/model/tokenizer";
import { transcribe } from "../src/model/transcribe";
import { loadWhisper, type WhisperConfig } from "../src/model/whisper";

/**
 * Head-to-head: kiln vs Transformers.js (ONNX Runtime Web, WebGPU backend)
 * running Whisper-tiny on the same audio in the same tab.
 *
 * Fairness rules:
 *  - same audio (LibriSpeech sample + the JFK clip), decoded once to 16 kHz
 *  - same scoring (src/eval/wer.ts)
 *  - greedy decoding, English, transcribe, no timestamps, both sides
 *  - each engine loaded and warmed up (shaders/sessions compiled) before
 *    timing; one engine at a time, the previous one released
 *  - the "int4" rows quantize the same part of the model on both sides:
 *    f32 encoder, 4-bit decoder
 */

const DIR = "/weights/whisper-tiny";
const EVAL = "/weights/eval/librispeech-dummy";
const JFK = "/weights/samples/jfk.wav";
const ONNX_MODEL = "onnx-community/whisper-tiny";

interface Utterance {
  file: string;
  text: string;
}

export interface EngineResult {
  engine: string;
  weights: string;
  wer: number;
  audioSeconds: number;
  totalMs: number;
  medianMs: number;
  jfkMs: number;
  jfkText: string;
}

type Transcriber = (samples: Float32Array) => Promise<string>;

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

async function measure(
  engine: string,
  weights: string,
  run: Transcriber,
  audio: Float32Array[],
  refs: string[],
  jfk: Float32Array,
  onStatus: (s: string) => void,
): Promise<EngineResult> {
  // warm-up: compile everything once on real input
  onStatus(`${engine} (${weights}): warming up…`);
  await run(audio[0]);
  await run(jfk);

  const times: number[] = [];
  const pairs: { reference: string; hypothesis: string }[] = [];
  for (const [i, a] of audio.entries()) {
    const t = performance.now();
    const text = await run(a);
    times.push(performance.now() - t);
    pairs.push({ reference: refs[i], hypothesis: text });
    if (i % 10 === 0) onStatus(`${engine} (${weights}): ${i + 1}/${audio.length}`);
  }
  const jfkRuns: number[] = [];
  let jfkText = "";
  for (let i = 0; i < 3; i++) {
    const t = performance.now();
    jfkText = await run(jfk);
    jfkRuns.push(performance.now() - t);
  }
  return {
    engine,
    weights,
    wer: corpusWer(pairs).wer,
    audioSeconds: audio.reduce((s, a) => s + a.length / 16000, 0),
    totalMs: times.reduce((a, b) => a + b, 0),
    medianMs: median(times),
    jfkMs: median(jfkRuns),
    jfkText,
  };
}

export async function runCompare(onStatus: (s: string) => void = () => {}, limit?: number): Promise<EngineResult[]> {
  onStatus("loading eval audio…");
  const manifest = ((await (await fetch(`${EVAL}/manifest.json`)).json()) as Utterance[]).slice(0, limit);
  const audio = await Promise.all(manifest.map(async (u) => decodeAudio(await (await fetch(`${EVAL}/${u.file}`)).arrayBuffer())));
  const refs = manifest.map((u) => u.text);
  const jfk = await decodeAudio(await (await fetch(JFK)).arrayBuffer());
  const results: EngineResult[] = [];

  // --- kiln ---------------------------------------------------------------------
  const kiln = await Kiln.create();
  const [weights, config, tokJson] = await Promise.all([
    fetch(`${DIR}/model.safetensors`).then((r) => r.arrayBuffer()),
    fetch(`${DIR}/config.json`).then((r) => r.json() as Promise<WhisperConfig>),
    fetch(`${DIR}/tokenizer.json`).then((r) => r.json()),
  ]);
  const tokenizer = new Tokenizer(tokJson);
  for (const q of ["f32", "int4"] as const) {
    onStatus(`kiln (${q}): loading…`);
    const model = loadWhisper(kiln, weights, config, { quantize: q === "int4" ? "int4" : undefined });
    const run: Transcriber = async (s) => (await transcribe(model, tokenizer, s)).text;
    results.push(await measure("kiln", q === "f32" ? "f32" : "f32 encoder, int4 decoder", run, audio, refs, jfk, onStatus));
    model.destroy();
  }
  kiln.destroy();

  // --- Transformers.js --------------------------------------------------------------
  const variants: { label: string; dtype: Record<string, string> }[] = [
    { label: "f32", dtype: { encoder_model: "fp32", decoder_model_merged: "fp32" } },
    { label: "f32 encoder, q4 decoder", dtype: { encoder_model: "fp32", decoder_model_merged: "q4" } },
  ];
  for (const v of variants) {
    onStatus(`Transformers.js (${v.label}): loading…`);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const asr: any = await pipeline("automatic-speech-recognition", ONNX_MODEL, { device: "webgpu", dtype: v.dtype as never });
    const run: Transcriber = async (s) => {
      const out = await asr(s, { language: "english", task: "transcribe", return_timestamps: false });
      return (Array.isArray(out) ? out[0].text : out.text).trim();
    };
    results.push(await measure("Transformers.js", v.label, run, audio, refs, jfk, onStatus));
    await asr.dispose?.();
  }

  onStatus("done");
  return results;
}

// --- page ---------------------------------------------------------------------------

const button = document.getElementById("run") as HTMLButtonElement | null;
if (button) {
  const status = document.getElementById("status")!;
  const table = document.getElementById("results")!;
  const tbody = table.querySelector("tbody")!;
  button.addEventListener("click", async () => {
    button.disabled = true;
    tbody.innerHTML = "";
    try {
      for (const r of await runCompare((s) => (status.textContent = s))) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${r.engine}</td><td>${r.weights}</td><td class="num">${(r.wer * 100).toFixed(2)}%</td>
          <td class="num">${r.audioSeconds.toFixed(0)} s</td><td class="num">${(r.totalMs / 1000).toFixed(2)} s</td>
          <td class="num">${(r.audioSeconds / (r.totalMs / 1000)).toFixed(0)}×</td>
          <td class="num">${r.medianMs.toFixed(0)} ms</td><td class="num">${r.jfkMs.toFixed(0)} ms</td>`;
        tbody.appendChild(tr);
      }
      table.hidden = false;
    } catch (e) {
      status.textContent = `error: ${(e as Error).message}`;
      console.error(e);
    } finally {
      button.disabled = false;
    }
  });
}

declare global {
  interface Window {
    runCompare: typeof runCompare;
  }
}
window.runCompare = runCompare;
