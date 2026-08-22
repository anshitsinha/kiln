import { describe, expect, it } from "vitest";
import { decodeAudio } from "../src/audio/decode";
import { encodeWav } from "../src/audio/wav";

describe("decodeAudio (Web Audio)", () => {
  it("resamples 44.1 kHz to 16 kHz and keeps the tone", async () => {
    const sr = 44100;
    const tone = Float32Array.from({ length: sr }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 440 * i) / sr));
    const out = await decodeAudio(encodeWav(tone, sr));
    expect(Math.abs(out.length - 16000)).toBeLessThanOrEqual(1);
    // amplitude survives the resample (peak ~0.5)
    const peak = out.slice(1000, 15000).reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    expect(peak).toBeGreaterThan(0.45);
    expect(peak).toBeLessThan(0.55);
    // and the frequency: count zero crossings over 1 s -> ~880
    let crossings = 0;
    for (let i = 1; i < out.length; i++) if (out[i - 1] < 0 !== out[i] < 0) crossings++;
    expect(crossings).toBeGreaterThan(860);
    expect(crossings).toBeLessThan(900);
  });

  it("passes 16 kHz through at the same length", async () => {
    const x = Float32Array.from({ length: 8000 }, (_, i) => Math.sin(i / 10) * 0.3);
    const out = await decodeAudio(encodeWav(x, 16000));
    expect(out.length).toBe(8000);
  });
});
