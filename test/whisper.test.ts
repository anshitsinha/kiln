import { describe, expect, it } from "vitest";
import { decodeWav } from "../src/audio/wav";
import { Tokenizer } from "../src/model/tokenizer";
import { transcribe } from "../src/model/transcribe";
import { loadWhisper, type WhisperConfig } from "../src/model/whisper";
import { getKiln, hasAsset } from "./helpers";

const DIR = "/weights/whisper-tiny";
const JFK = "/weights/samples/jfk.wav";
const ready = (await hasAsset(`${DIR}/model.safetensors`)) && (await hasAsset(JFK));

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
});
