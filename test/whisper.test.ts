import { describe, expect, it } from "vitest";
import { decodeWav } from "../src/audio/wav";
import { Tokenizer } from "../src/model/tokenizer";
import { transcribe } from "../src/model/transcribe";
import { loadWhisper, type WhisperConfig } from "../src/model/whisper";
import { getKiln, hasAsset } from "./helpers";

const DIR = "/weights/whisper-tiny";
const JFK = "/weights/samples/jfk.wav";
// whisper-tiny (4 + 4 layers)
const DECODE_STEP_BUDGET = 52;
const ENCODE_BUDGET = 37;
const ready =(await hasAsset(`${DIR}/model.safetensors`)) && (await hasAsset(JFK));

describe("whisper-tiny end to end", () => {
  // The real test of every kernel at once: if any of them is off, the
  // transcript turns to garbage. Needs `npm run fetch`.
  it.runIf(ready)("transcribes JFK's inaugural address clip", { timeout: 120_000 }, async () => {
    const kiln = await getKiln();
    const [weights, config, tokJson, wav] = await Promise.all([
      fetch(`${DIR}/model.safetensors`).then((r) => r.arrayBuffer()),
      fetch(`${DIR}/config.json`).then((r) => r.json() as Promise<WhisperConfig>),
      fetch(`${DIR}/tokenizer.json`).then((r) => r.json()),
      fetch(JFK).then((r) => r.arrayBuffer()),
    ]);
    const model = loadWhisper(kiln, weights, config);
    const tokenizer = new Tokenizer(tokJson);
    const { samples } = decodeWav(wav);

    const result = await transcribe(model, tokenizer, samples);
    console.log(JSON.stringify(result.text), result.timings);

    const words = result.text.toLowerCase().replace(/[^a-z ]/g, "");
    expect(words).toContain("ask not what your country can do for you");
    expect(words).toContain("ask what you can do for your country");

    // a second, warm run (shaders compiled) must give the same answer and
    // not leak: everything per-transcription is scoped or freed
    const warm = await transcribe(model, tokenizer, samples);
    console.log("warm:", warm.timings);
    expect(warm.text).toBe(result.text);

    // the CPU reference spectrogram must give the same transcript
    const cpu = await transcribe(model, tokenizer, samples, { cpuMel: true });
    expect(cpu.text).toBe(result.text);
  });

  // Dispatch budget: on WebGPU a decode step's cost is mostly the number of
  // kernel launches, so pin it down - a refactor that quietly adds kernels
  // to the per-token path should fail here, not just get slower.
  it.runIf(ready)("dispatches per encode and per decode step", { timeout: 120_000 }, async () => {
    const kiln = await getKiln();
    const [weights, config] = await Promise.all([
      fetch(`${DIR}/model.safetensors`).then((r) => r.arrayBuffer()),
      fetch(`${DIR}/config.json`).then((r) => r.json() as Promise<WhisperConfig>),
    ]);
    const model = loadWhisper(kiln, weights, config);
    const mel = new Float32Array(config.num_mel_bins * 3000);
    const count = async (fn: () => Promise<unknown>) => {
      const before = kiln.dispatchCount;
      await fn();
      return kiln.dispatchCount - before;
    };
    await kiln.scope(async () => {
      const enc = await model.encode(mel);
      const encDispatches = await count(() => model.encode(mel).then((t) => t.destroy()));
      const state = await model.startDecode(enc);
      await model.decodeNext(state, [50258, 50259, 50359, 50363]); // prompt
      const step = await count(() => model.decodeNext(state, [440]));
      // warm timing over a few steps
      const t0 = performance.now();
      const STEPS = 20;
      for (let i = 0; i < STEPS; i++) await model.decodeNext(state, [440]);
      const ms = (performance.now() - t0) / STEPS;
      // GPU time alone (timestamp queries), when the device supports them
      let gpu = "";
      if (kiln.features.timestamps) {
        const t = await kiln.profile(async () => void (await model.decodeNext(state, [440])));
        gpu = `, ${t.reduce((a, k) => a + k.ms, 0).toFixed(2)} ms of it on the GPU`;
      }
      console.log(`encode: ${encDispatches} dispatches; decode step: ${step} dispatches, ${ms.toFixed(2)} ms${gpu}`);
      expect(step).toBeLessThanOrEqual(DECODE_STEP_BUDGET);
      expect(encDispatches).toBeLessThanOrEqual(ENCODE_BUDGET);
      model.freeDecode(state);
    });
    model.destroy();
  });

  // int4 decoder: the quantization error is real, so this checks the
  // transcript is still right rather than identical to f32
  it.runIf(ready)("transcribes with an int4 decoder", { timeout: 120_000 }, async () => {
    const kiln = await getKiln();
    const [weights, config, tokJson, wav] = await Promise.all([
      fetch(`${DIR}/model.safetensors`).then((r) => r.arrayBuffer()),
      fetch(`${DIR}/config.json`).then((r) => r.json() as Promise<WhisperConfig>),
      fetch(`${DIR}/tokenizer.json`).then((r) => r.json()),
      fetch(JFK).then((r) => r.arrayBuffer()),
    ]);
    const model = loadWhisper(kiln, weights, config, { quantize: "int4" });
    const tokenizer = new Tokenizer(tokJson);
    const { samples } = decodeWav(wav);
    await transcribe(model, tokenizer, samples); // warm up
    const result = await transcribe(model, tokenizer, samples);
    console.log("int4:", JSON.stringify(result.text), result.timings);
    const words = result.text.toLowerCase().replace(/[^a-z ]/g, "");
    expect(words).toContain("ask not what your country can do for you");
  });
});
