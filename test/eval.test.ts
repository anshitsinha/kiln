import { describe, expect, it } from "vitest";
import { decodeAudio } from "../src/audio/decode";
import { corpusWer, wordErrors } from "../src/eval/wer";
import { Tokenizer } from "../src/model/tokenizer";
import { transcribe, transcribeLong } from "../src/model/transcribe";
import { loadWhisper, type Whisper, type WhisperConfig } from "../src/model/whisper";
import { getKiln, hasAsset } from "./helpers";

const DIR = "/weights/whisper-tiny";
const EVAL = "/weights/eval/librispeech-dummy";
const ready = (await hasAsset(`${DIR}/model.safetensors`)) && (await hasAsset(`${EVAL}/manifest.json`));

interface Utterance {
  id: string;
  file: string;
  text: string;
}

async function setup() {
  const kiln = await getKiln();
  const [weights, config, tokJson, manifest] = await Promise.all([
    fetch(`${DIR}/model.safetensors`).then((r) => r.arrayBuffer()),
    fetch(`${DIR}/config.json`).then((r) => r.json() as Promise<WhisperConfig>),
    fetch(`${DIR}/tokenizer.json`).then((r) => r.json()),
    fetch(`${EVAL}/manifest.json`).then((r) => r.json() as Promise<Utterance[]>),
  ]);
  const audio = await Promise.all(manifest.map(async (u) => decodeAudio(await (await fetch(`${EVAL}/${u.file}`)).arrayBuffer())));
  return { kiln, weights, config, tokenizer: new Tokenizer(tokJson), manifest, audio };
}

async function evaluate(model: Whisper, tokenizer: Tokenizer, manifest: Utterance[], audio: Float32Array[]) {
  const pairs: { reference: string; hypothesis: string }[] = [];
  let ms = 0;
  let seconds = 0;
  for (const [i, u] of manifest.entries()) {
    const r = await transcribe(model, tokenizer, audio[i]);
    pairs.push({ reference: u.text, hypothesis: r.text });
    ms += r.timings.melMs + r.timings.encodeMs + r.timings.decodeMs;
    seconds += audio[i].length / 16000;
  }
  return { ...corpusWer(pairs), pairs, ms, seconds };
}

describe("LibriSpeech eval (73 utterances)", () => {
  it.runIf(ready)("f32 vs int4 word error rate", { timeout: 600_000 }, async () => {
    const { kiln, weights, config, tokenizer, manifest, audio } = await setup();
    const results: Record<string, Awaited<ReturnType<typeof evaluate>>> = {};
    for (const precision of ["f32", "int4"] as const) {
      const model = loadWhisper(kiln, weights, config, { quantize: precision === "int4" ? "int4" : undefined });
      await transcribe(model, tokenizer, audio[0]); // warm up shaders
      results[precision] = await evaluate(model, tokenizer, manifest, audio);
      model.destroy();
      const r = results[precision];
      console.log(
        `${precision}: WER ${(r.wer * 100).toFixed(2)}% (S ${r.substitutions}, D ${r.deletions}, I ${r.insertions} / ${r.refWords} words), ` +
          `${r.seconds.toFixed(0)} s of audio in ${(r.ms / 1000).toFixed(1)} s (${(r.seconds / (r.ms / 1000)).toFixed(0)}x real time)`,
      );
    }
    // utterances where int4 makes more errors than f32
    const worse = results.f32.pairs.filter((p, i) => wordErrors(p.reference, results.int4.pairs[i].hypothesis).wer > wordErrors(p.reference, p.hypothesis).wer);
    console.log(`int4 worse than f32 on ${worse.length}/${manifest.length} utterances`);
    // Regression guards, not targets: whisper-tiny's published LibriSpeech
    // test-clean WER is ~7.5% with Whisper's full normaliser; this is a
    // different (validation) sample with a simpler one.
    expect(results.f32.wer).toBeLessThan(0.13);
    expect(results.int4.wer).toBeLessThan(results.f32.wer + 0.04);
  });

  it.runIf(ready)("long-form: ~70 s of joined utterances, chunked at pauses", { timeout: 300_000 }, async () => {
    const { kiln, weights, config, tokenizer, manifest, audio } = await setup();
    const model = loadWhisper(kiln, weights, config);
    // join utterances with 0.6 s pauses until past 60 s
    const parts: Float32Array[] = [];
    const refs: string[] = [];
    let total = 0;
    for (let i = 0; total < 60 * 16000; i++) {
      parts.push(audio[i], new Float32Array(Math.round(0.6 * 16000)));
      refs.push(manifest[i].text);
      total += audio[i].length + Math.round(0.6 * 16000);
    }
    const long = new Float32Array(total);
    let o = 0;
    for (const p of parts) {
      long.set(p, o);
      o += p.length;
    }
    const r = await transcribeLong(model, tokenizer, long);
    const wer = wordErrors(refs.join(" "), r.text);
    console.log(`long-form: ${(total / 16000).toFixed(1)} s in ${r.chunks} chunks, WER ${(wer.wer * 100).toFixed(2)}%`);
    expect(r.chunks).toBeGreaterThanOrEqual(3);
    expect(wer.wer).toBeLessThan(0.15);
    model.destroy();
  });
});
