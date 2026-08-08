import { logMelSpectrogram } from "../audio/mel";
import type { Tokenizer } from "./tokenizer";
import type { Whisper } from "./whisper";

export interface TranscribeOptions {
  language?: string; // "en", "de", ... -> <|en|>
  maxTokens?: number;
  /** called with the text so far after every token */
  onToken?: (text: string) => void;
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

  let t = performance.now();
  const mel = logMelSpectrogram(samples);
  const melMs = performance.now() - t;

  t = performance.now();
  const enc = await model.encode(mel);
  const state = await model.startDecode(enc);
  enc.destroy(); // cross-attention K/V are all the decoder needs from it
  const encodeMs = performance.now() - t;

  t = performance.now();
  const out: number[] = [];
  try {
    let logits = await model.decode(state, prompt);
    for (let i = 0; i < maxTokens; i++) {
      const next = argmaxText(logits, eot);
      if (next === eot) break;
      out.push(next);
      opts.onToken?.(tokenizer.decode(out));
      logits = await model.decode(state, [next]);
    }
  } finally {
    model.freeDecode(state);
  }
  const decodeMs = performance.now() - t;

  return { text: tokenizer.decode(out).trim(), tokens: out, timings: { melMs, encodeMs, decodeMs, tokens: out.length } };
}

/** argmax over the text vocabulary (ids below eot) plus eot itself. */
function argmaxText(logits: Float32Array, eot: number): number {
  let best = eot;
  for (let i = 0; i < eot; i++) if (logits[i] > logits[best]) best = i;
  return best;
}
