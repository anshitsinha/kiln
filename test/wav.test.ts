import { describe, expect, it } from "vitest";
import { decodeWav, encodeWav, resample } from "../src/audio/wav";
import { hasAsset } from "./helpers";

const JFK = "/weights/samples/jfk.wav";
const hasJfk = await hasAsset(JFK);

describe("wav", () => {
  it("round-trips 16-bit PCM", () => {
    const samples = Float32Array.from({ length: 100 }, (_, i) => Math.sin(i / 5) * 0.8);
    const { samples: back, sampleRate } = decodeWav(encodeWav(samples, 16000));
    expect(sampleRate).toBe(16000);
    expect(back.length).toBe(100);
    for (let i = 0; i < 100; i++) expect(back[i]).toBeCloseTo(samples[i], 4);
  });

  it("resamples 48 kHz to 16 kHz", () => {
    const tone = Float32Array.from({ length: 48000 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / 48000));
    const out = resample({ samples: tone, sampleRate: 48000 }, 16000);
    expect(out.length).toBe(16000);
    // sample 1000 at 16 kHz is t = 1000/16000 s
    expect(out[1000]).toBeCloseTo(Math.sin((2 * Math.PI * 440 * 1000) / 16000), 2);
  });

  it("rejects non-WAV data", () => {
    expect(() => decodeWav(new ArrayBuffer(64))).toThrow(/RIFF/);
  });

  // needs `npm run fetch`; shows up as skipped (not silently passing) without it
  it.runIf(hasJfk)("decodes the JFK sample", async () => {
    const res = await fetch(JFK);
    const { samples, sampleRate } = decodeWav(await res.arrayBuffer());
    expect(sampleRate).toBe(16000);
    expect(samples.length / 16000).toBeCloseTo(11, 0); // ~11 seconds
  });
});
