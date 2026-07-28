import { rfft } from "./fft";

/**
 * Whisper's log-mel spectrogram, matching openai/whisper's audio.py:
 *
 *   - pad / trim to 30 s (480,000 samples at 16 kHz)
 *   - STFT: n_fft 400, hop 160, periodic Hann window, centred with reflect
 *     padding (torch.stft(center=True)), power spectrum, last frame dropped
 *     -> 3000 frames
 *   - 80 mel bands, librosa's Slaney-style filterbank (slaney mel scale,
 *     slaney area normalisation)
 *   - log10(max(mel, 1e-10)), floored at (global max - 8), then (x + 4) / 4
 *
 * Output is [80, 3000], mel band major - the layout the encoder's first
 * conv expects (channels x time).
 */

export const SAMPLE_RATE = 16000;
export const N_FFT = 400;
export const HOP = 160;
export const N_MELS = 80;
export const CHUNK_SAMPLES = 30 * SAMPLE_RATE;
export const N_FRAMES = CHUNK_SAMPLES / HOP; // 3000

// --- Slaney mel scale (librosa htk=False) ---------------------------------------

const F_SP = 200 / 3; // Hz per mel below 1 kHz
const MIN_LOG_HZ = 1000;
const MIN_LOG_MEL = MIN_LOG_HZ / F_SP; // 15
const LOGSTEP = Math.log(6.4) / 27;

export function hzToMel(f: number): number {
  return f < MIN_LOG_HZ ? f / F_SP : MIN_LOG_MEL + Math.log(f / MIN_LOG_HZ) / LOGSTEP;
}

export function melToHz(m: number): number {
  return m < MIN_LOG_MEL ? m * F_SP : MIN_LOG_HZ * Math.exp(LOGSTEP * (m - MIN_LOG_MEL));
}

/**
 * [nMels, nFft/2 + 1] triangular filters, librosa.filters.mel(sr, n_fft,
 * n_mels) with default (slaney) normalisation: each triangle is scaled by
 * 2 / (its width in Hz), so filters have roughly equal area.
 */
export function melFilters(sr = SAMPLE_RATE, nFft = N_FFT, nMels = N_MELS): Float32Array {
  const nBins = nFft / 2 + 1;
  const fftFreqs = Float64Array.from({ length: nBins }, (_, k) => (k * sr) / nFft);
  const melMin = hzToMel(0);
  const melMax = hzToMel(sr / 2);
  const pts = Float64Array.from({ length: nMels + 2 }, (_, i) => melToHz(melMin + ((melMax - melMin) * i) / (nMels + 1)));

  const out = new Float32Array(nMels * nBins);
  for (let m = 0; m < nMels; m++) {
    const [lo, mid, hi] = [pts[m], pts[m + 1], pts[m + 2]];
    const enorm = 2 / (hi - lo);
    for (let k = 0; k < nBins; k++) {
      const up = (fftFreqs[k] - lo) / (mid - lo);
      const down = (hi - fftFreqs[k]) / (hi - mid);
      out[m * nBins + k] = Math.max(0, Math.min(up, down)) * enorm;
    }
  }
  return out;
}

/** Periodic Hann window (torch.hann_window default). */
function hann(n: number): Float64Array {
  return Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
}

/** numpy/torch "reflect" padding: mirror without repeating the edge sample. */
function reflectPad(x: Float32Array, pad: number): Float32Array {
  const out = new Float32Array(x.length + 2 * pad);
  out.set(x, pad);
  for (let i = 0; i < pad; i++) {
    out[pad - 1 - i] = x[Math.min(i + 1, x.length - 1)];
    out[pad + x.length + i] = x[Math.max(x.length - 2 - i, 0)];
  }
  return out;
}

let filtersCache: Float32Array | null = null;

/** Log-mel spectrogram of up to 30 s of 16 kHz mono audio -> [80, 3000]. */
export function logMelSpectrogram(samples: Float32Array): Float32Array {
  const audio = new Float32Array(CHUNK_SAMPLES);
  audio.set(samples.subarray(0, CHUNK_SAMPLES));

  const padded = reflectPad(audio, N_FFT / 2);
  const window = hann(N_FFT);
  const nBins = N_FFT / 2 + 1;
  filtersCache ??= melFilters();
  const filters = filtersCache;

  const mel = new Float32Array(N_MELS * N_FRAMES);
  const frame = new Float64Array(N_FFT);
  const power = new Float64Array(nBins);
  for (let t = 0; t < N_FRAMES; t++) {
    for (let i = 0; i < N_FFT; i++) frame[i] = padded[t * HOP + i] * window[i];
    const { re, im } = rfft(frame);
    for (let k = 0; k < nBins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      const row = m * nBins;
      for (let k = 0; k < nBins; k++) s += filters[row + k] * power[k];
      mel[m * N_FRAMES + t] = s;
    }
  }

  let max = -Infinity;
  for (let i = 0; i < mel.length; i++) {
    mel[i] = Math.log10(Math.max(mel[i], 1e-10));
    if (mel[i] > max) max = mel[i];
  }
  for (let i = 0; i < mel.length; i++) mel[i] = (Math.max(mel[i], max - 8) + 4) / 4;
  return mel;
}
