/**
 * Splitting long audio into Whisper-sized (<= 30 s) pieces.
 *
 * Whisper's reference long-form decoding predicts timestamps and seeks to
 * the end of the last complete segment. This is the simpler alternative a
 * lot of production pipelines use: cut each window at the quietest moment
 * near its end, so the cut lands in a pause between words rather than in
 * the middle of one. It keeps the fast no-timestamps decoding path, at the
 * cost of occasionally splitting a word when someone talks for 30 s without
 * pausing.
 */

export interface Chunk {
  start: number; // sample index, inclusive
  end: number; // exclusive
}

export interface SplitOptions {
  sampleRate?: number;
  maxSeconds?: number;
  /** how far back from the limit to look for a pause */
  searchSeconds?: number;
}

export function splitOnSilence(samples: Float32Array, opts: SplitOptions = {}): Chunk[] {
  const sr = opts.sampleRate ?? 16000;
  const max = Math.floor((opts.maxSeconds ?? 30) * sr);
  const search = Math.floor((opts.searchSeconds ?? 5) * sr);
  const frame = Math.floor(0.02 * sr); // 20 ms energy frames
  const hop = Math.floor(frame / 2);

  const chunks: Chunk[] = [];
  let start = 0;
  while (samples.length - start > max) {
    // quietest 20 ms frame in [start + max - search, start + max)
    let bestCut = start + max;
    let bestEnergy = Infinity;
    for (let f = start + max - search; f + frame <= start + max; f += hop) {
      let e = 0;
      for (let i = f; i < f + frame; i++) e += samples[i] * samples[i];
      if (e < bestEnergy) {
        bestEnergy = e;
        bestCut = f + (frame >> 1);
      }
    }
    chunks.push({ start, end: bestCut });
    start = bestCut;
  }
  if (start < samples.length || chunks.length === 0) chunks.push({ start, end: samples.length });
  return chunks;
}
