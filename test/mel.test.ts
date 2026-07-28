import { describe, expect, it } from "vitest";
import { hzToMel, logMelSpectrogram, melFilters, melToHz, N_FFT, N_FRAMES, N_MELS, SAMPLE_RATE } from "../src/audio/mel";
import { decodeWav } from "../src/audio/wav";
import { hasAsset } from "./helpers";

const JFK = "/weights/samples/jfk.wav";
const hasJfk = await hasAsset(JFK);
const nBins = N_FFT / 2 + 1;

describe("mel scale + filterbank", () => {
  it("uses the Slaney scale (linear below 1 kHz, log above)", () => {
    expect(hzToMel(1000)).toBeCloseTo(15, 10);
    expect(hzToMel(500)).toBeCloseTo(7.5, 10);
    for (const f of [0, 300, 1000, 2500, 8000]) expect(melToHz(hzToMel(f))).toBeCloseTo(f, 6);
  });

  it("builds 80 triangles that tile the spectrum", () => {
    const f = melFilters();
    expect(f.length).toBe(N_MELS * nBins);
    for (let m = 0; m < N_MELS; m++) {
      const row = f.subarray(m * nBins, (m + 1) * nBins);
      // non-negative, single peak (rises then falls)
      expect(row.every((v) => v >= 0)).toBe(true);
      const peak = row.indexOf(Math.max(...row));
      for (let k = 1; k <= peak; k++) expect(row[k]).toBeGreaterThanOrEqual(row[k - 1]);
      for (let k = peak + 1; k < nBins; k++) expect(row[k]).toBeLessThanOrEqual(row[k - 1]);
    }
    // slaney normalisation: area in Hz ~ 1 once a filter spans several bins
    const binHz = SAMPLE_RATE / N_FFT;
    for (const m of [40, 60, 79]) {
      const area = f.subarray(m * nBins, (m + 1) * nBins).reduce((a, b) => a + b) * binHz;
      expect(area).toBeGreaterThan(0.9);
      expect(area).toBeLessThan(1.1);
    }
  });
});

describe("log-mel spectrogram", () => {
  it("is constant for silence: log10(1e-10) floored then scaled -> -1.5", () => {
    const mel = logMelSpectrogram(new Float32Array(16000));
    expect(mel.length).toBe(N_MELS * N_FRAMES);
    expect(mel.every((v) => Math.abs(v - -1.5) < 1e-6)).toBe(true);
  });

  it("puts a 1 kHz tone in the band around 1 kHz", () => {
    const tone = Float32Array.from({ length: SAMPLE_RATE }, (_, i) => Math.sin((2 * Math.PI * 1000 * i) / SAMPLE_RATE));
    const mel = logMelSpectrogram(tone);
    // look at a frame in the middle of the tone
    const t = 50;
    let best = 0;
    for (let m = 0; m < N_MELS; m++) if (mel[m * N_FRAMES + t] > mel[best * N_FRAMES + t]) best = m;
    // band m is centred on melToHz(mel(8000) * (m + 1) / 81)
    const centre = melToHz((hzToMel(8000) * (best + 1)) / (N_MELS + 1));
    expect(Math.abs(centre - 1000)).toBeLessThan(80);
  });

  it.runIf(hasJfk)("has the expected range on real speech", async () => {
    const { samples } = decodeWav(await (await fetch(JFK)).arrayBuffer());
    const mel = logMelSpectrogram(samples);
    let min = Infinity;
    let max = -Infinity;
    for (const v of mel) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    // max - 8 floor then /4 => the spread is at most 2; max is (max_log10 + 4) / 4
    expect(max - min).toBeLessThanOrEqual(2 + 1e-6);
    expect(max).toBeGreaterThan(0.5);
    expect(max).toBeLessThan(2.5);
  });
});
