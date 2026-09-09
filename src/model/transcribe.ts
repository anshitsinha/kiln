import { splitOnSilence } from "../audio/chunk";
import { logMelSpectrogram, SAMPLE_RATE } from "../audio/mel";
import { logMelSpectrogramGpu } from "../audio/mel-gpu";
import type { Tokenizer } from "./tokenizer";
import type { Whisper } from "./whisper";

export interface TranscribeOptions {
  language?: string; // "en", "de", ... -> <|en|>
  maxTokens?: number;
  /** called with the text so far after every token */
  onToken?: (text: string) => void;
  /** use the CPU (f64 FFT) spectrogram instead of the GPU one */
  cpuMel?: boolean;
}

export interface Transcript {
  text: string;
  tokens: number[];
  timings: { melMs: number; encodeMs: number; decodeMs: number; tokens: number };
}

/**
 * Greedy transcription of up to 30 s of 16 kHz mono audio.
 *
 * Prompt: <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>,
 * then repeatedly take the most likely next token until <|endoftext|>.
 * Only text tokens and end-of-text are allowed - Whisper's other specials
 * (language tags, timestamps, task tokens) can't appear mid-transcript.
 */
export async function transcribe(
  model: Whisper,
  tokenizer: Tokenizer,
  samples: Float32Array,
  opts: TranscribeOptions = {},
): Promise<Transcript> {
  const eot = tokenizer.id("<|endoftext|>");
  const prompt = [
    tokenizer.id("<|startoftranscript|>"),
    tokenizer.id(`<|${opts.language ?? "en"}|>`),
    tokenizer.id("<|transcribe|>"),
    tokenizer.id("<|notimestamps|>"),
  ];
  const maxTokens = Math.min(opts.maxTokens ?? 224, model.config.max_target_positions - prompt.length);

  // GPU spectrogram by default; the CPU one is the reference implementation
  let t = performance.now();
  const kiln = model.kiln;
  const mel = opts.cpuMel ? logMelSpectrogram(samples) : logMelSpectrogramGpu(kiln, samples);
  if (!opts.cpuMel) await kiln.flush(); // so melMs measures the GPU work, not just recording it
  const melMs = performance.now() - t;

  t = performance.now();
  const enc = await model.encode(mel);
  if (!(mel instanceof Float32Array)) mel.destroy();
  const state = await model.startDecode(enc);
  enc.destroy(); // cross-attention K/V are all the decoder needs from it
  const encodeMs = performance.now() - t;

  t = performance.now();
  const out: number[] = [];
  // greedy choice among text tokens (ids below eot) plus eot itself, done on
  // the GPU so each step reads back one id
  const choice = { limit: eot, extra: eot };
  try {
    let next = await model.decodeNext(state, prompt, choice);
    for (let i = 0; i < maxTokens && next !== eot; i++) {
      out.push(next);
      opts.onToken?.(tokenizer.decode(out));
      next = await model.decodeNext(state, [next], choice);
    }
  } finally {
    model.freeDecode(state);
  }
  const decodeMs = performance.now() - t;

  return { text: tokenizer.decode(out).trim(), tokens: out, timings: { melMs, encodeMs, decodeMs, tokens: out.length } };
}

/**
 * Any length of audio: split into <= 30 s chunks at pauses (see
 * splitOnSilence), transcribe each, join the text. Timings are summed.
 */
export async function transcribeLong(
  model: Whisper,
  tokenizer: Tokenizer,
  samples: Float32Array,
  opts: TranscribeOptions & { onChunk?: (index: number, count: number) => void } = {},
): Promise<Transcript & { chunks: number }> {
  const chunks = splitOnSilence(samples, { sampleRate: SAMPLE_RATE });
  const texts: string[] = [];
  const tokens: number[] = [];
  const timings = { melMs: 0, encodeMs: 0, decodeMs: 0, tokens: 0 };
  for (const [i, c] of chunks.entries()) {
    opts.onChunk?.(i, chunks.length);
    const prefix = texts.join(" ");
    const r = await transcribe(model, tokenizer, samples.subarray(c.start, c.end), {
      ...opts,
      onToken: opts.onToken && ((t) => opts.onToken!(`${prefix} ${t}`.trim())),
    });
    if (r.text) texts.push(r.text);
    tokens.push(...r.tokens);
    timings.melMs += r.timings.melMs;
    timings.encodeMs += r.timings.encodeMs;
    timings.decodeMs += r.timings.decodeMs;
    timings.tokens += r.timings.tokens;
  }
  return { text: texts.join(" "), tokens, timings, chunks: chunks.length };
}
