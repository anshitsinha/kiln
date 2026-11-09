import { Kiln } from "../src";
import { decodeAudio } from "../src/audio/decode";
import { SAMPLE_RATE } from "../src/audio/mel";
import { fetchCached, hubUrl } from "../src/model/hub";
import { Tokenizer } from "../src/model/tokenizer";
import { transcribeLong } from "../src/model/transcribe";
import { loadWhisper, type Whisper, type WhisperConfig } from "../src/model/whisper";

const MODELS = {
  tiny: { repo: "openai/whisper-tiny", mb: 151 },
  base: { repo: "openai/whisper-base", mb: 290 },
} as const;
type Size = keyof typeof MODELS;
const SAMPLE_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const loadBtn = $<HTMLButtonElement>("load");
const precision = $<HTMLSelectElement>("precision");
const sizeSel = $<HTMLSelectElement>("size");
const sampleBtn = $<HTMLButtonElement>("sample");
const upload = $<HTMLInputElement>("upload");
const uploadLabel = $("uploadLabel");
const recordBtn = $<HTMLButtonElement>("record");
const bar = $("bar");
const modelStatus = $("modelStatus");
const audioStatus = $("audioStatus");
const transcriptEl = $("transcript");

let kiln: Kiln | null = null;
let model: Whisper | null = null;
let tokenizer: Tokenizer | null = null;
let loadedChoice = ""; // "<size>/<precision>" of the model on the GPU
let busy = false;

const choice = () => `${sizeSel.value}/${precision.value}`;
const loadLabel = () => {
  const m = MODELS[sizeSel.value as Size];
  return `Load whisper-${sizeSel.value} (${m.mb} MB, cached after the first time)`;
};

const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;
const ms = (n: number) => (n < 1000 ? `${Math.round(n)} ms` : `${(n / 1000).toFixed(2)} s`);

function setAudioEnabled(on: boolean) {
  sampleBtn.disabled = !on;
  upload.disabled = !on;
  recordBtn.disabled = !on;
  uploadLabel.setAttribute("aria-disabled", String(!on));
}

// --- model ----------------------------------------------------------------------

async function loadModel() {
  if (busy) return;
  busy = true;
  loadBtn.disabled = true;
  setAudioEnabled(false);
  const progress = bar.parentElement!;
  progress.hidden = false;
  try {
    kiln ??= await Kiln.create();
    modelStatus.textContent = `GPU: ${kiln.adapterName}`;
    const size = sizeSel.value as Size;
    const REPO = MODELS[size].repo;

    const [configBuf, tokBuf] = await Promise.all([
      fetchCached(hubUrl(REPO, "config.json")),
      fetchCached(hubUrl(REPO, "tokenizer.json")),
    ]);
    const config = JSON.parse(new TextDecoder().decode(configBuf)) as WhisperConfig;
    tokenizer = new Tokenizer(JSON.parse(new TextDecoder().decode(tokBuf)));

    const weights = await fetchCached(hubUrl(REPO, "model.safetensors"), ({ loaded, total, cached }) => {
      bar.style.width = total ? `${(100 * loaded) / total}%` : "50%";
      modelStatus.textContent = cached ? "Loaded weights from the browser cache" : `Downloading weights… ${mb(loaded)}${total ? ` / ${mb(total)}` : ""}`;
    });

    modelStatus.textContent = precision.value === "int4" ? "Quantizing the decoder to int4 and uploading to the GPU…" : "Uploading weights to the GPU…";
    await new Promise((r) => setTimeout(r, 30)); // let the status paint
    const t = performance.now();
    model?.destroy(); // free the previous precision's weights before uploading new ones
    model = loadWhisper(kiln, weights, config, { quantize: precision.value === "int4" ? "int4" : undefined });
    await kiln.flush();
    loadedChoice = choice();
    modelStatus.textContent =
      `Ready · whisper-${size} · ${precision.value} · ${mb(model.weightBytes)} on the GPU · ${kiln.adapterName} · ` +
      `set up in ${ms(performance.now() - t)}`;
    loadBtn.textContent = "Model loaded";
    setAudioEnabled(true);
  } catch (e) {
    modelStatus.textContent = `Couldn't load the model: ${(e as Error).message}`;
    loadBtn.disabled = false;
  } finally {
    progress.hidden = true;
    busy = false;
  }
}

loadBtn.addEventListener("click", loadModel);
// switching model or precision needs a reload of the weights
const onChoice = () => {
  if (!model) {
    loadBtn.textContent = loadLabel();
  } else if (choice() !== loadedChoice) {
    loadBtn.disabled = false;
    loadBtn.textContent = `Reload as whisper-${sizeSel.value} · ${precision.value}`;
  } else {
    loadBtn.disabled = true;
    loadBtn.textContent = "Model loaded";
  }
};
precision.addEventListener("change", onChoice);
sizeSel.addEventListener("change", onChoice);

// --- transcription -------------------------------------------------------------------

async function run(data: ArrayBuffer, label: string) {
  if (!model || !tokenizer || busy) return;
  busy = true;
  setAudioEnabled(false);
  try {
    audioStatus.textContent = `Decoding ${label}…`;
    const samples = await decodeAudio(data, SAMPLE_RATE);
    const seconds = samples.length / SAMPLE_RATE;
    audioStatus.textContent = `${label}: ${seconds.toFixed(1)} s`;

    transcriptEl.textContent = "…";
    // anything over 30 s is split at pauses and transcribed chunk by chunk
    const result = await transcribeLong(model, tokenizer, samples, {
      onToken: (text) => (transcriptEl.textContent = text.trim() || "…"),
      onChunk: (i, n) => {
        if (n > 1) audioStatus.textContent = `${label}: ${seconds.toFixed(1)} s · chunk ${i + 1} of ${n}`;
      },
    });
    transcriptEl.textContent = result.text || "(no speech detected)";
    if (result.chunks > 1) audioStatus.textContent = `${label}: ${seconds.toFixed(1)} s · ${result.chunks} chunks`;

    const { melMs, encodeMs, decodeMs, tokens } = result.timings;
    const audioS = seconds;
    const total = melMs + encodeMs + decodeMs;
    $("stats").hidden = false;
    $("sAudio").textContent = `${audioS.toFixed(1)} s`;
    $("sMel").textContent = ms(melMs);
    $("sEnc").textContent = ms(encodeMs);
    $("sDec").textContent = `${ms(decodeMs)} · ${tokens} tok`;
    $("sRtf").textContent = `${(audioS / (total / 1000)).toFixed(0)}× real time`;
  } catch (e) {
    transcriptEl.textContent = "";
    audioStatus.textContent = `Something went wrong: ${(e as Error).message}`;
  } finally {
    busy = false;
    setAudioEnabled(true);
  }
}

sampleBtn.addEventListener("click", async () => {
  audioStatus.textContent = "Fetching the sample…";
  const res = await fetch(SAMPLE_URL);
  await run(await res.arrayBuffer(), "JFK sample");
});

upload.addEventListener("change", async () => {
  const file = upload.files?.[0];
  upload.value = "";
  if (file) await run(await file.arrayBuffer(), file.name);
});

// --- recording --------------------------------------------------------------------

let recorder: MediaRecorder | null = null;
let stopTimer = 0;

recordBtn.addEventListener("click", async () => {
  if (recorder) {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks: Blob[] = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      clearTimeout(stopTimer);
      stream.getTracks().forEach((t) => t.stop());
      recorder = null;
      recordBtn.classList.remove("recording");
      recordBtn.textContent = "● Record";
      const blob = new Blob(chunks, { type: chunks[0]?.type });
      await run(await blob.arrayBuffer(), "recording");
    };
    recorder.start();
    recordBtn.classList.add("recording");
    recordBtn.textContent = "■ Stop";
    audioStatus.textContent = "Recording… (stops automatically at 5 minutes)";
    stopTimer = window.setTimeout(() => recorder?.stop(), 5 * 60_000);
  } catch (e) {
    audioStatus.textContent = `Microphone unavailable: ${(e as Error).message}`;
  }
});

// --- startup ------------------------------------------------------------------------

if (!("gpu" in navigator)) {
  $("nogpu").hidden = false;
  loadBtn.disabled = true;
}

// for automated runs
declare global {
  interface Window {
    kilnDemo: { load: typeof loadModel; transcribeUrl: (url: string) => Promise<string> };
  }
}
window.kilnDemo = {
  load: loadModel,
  transcribeUrl: async (url) => {
    await run(await (await fetch(url)).arrayBuffer(), url);
    return transcriptEl.textContent ?? "";
  },
};
