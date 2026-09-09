import { describe, expect, it } from "vitest";
import { splitOnSilence } from "../src/audio/chunk";

const SR = 16000;
const tone = (seconds: number) => Float32Array.from({ length: seconds * SR }, (_, i) => 0.5 * Math.sin(i / 7));
const silence = (seconds: number) => new Float32Array(Math.round(seconds * SR));
const concat = (...parts: Float32Array[]) => {
  const out = new Float32Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
};

describe("splitOnSilence", () => {
  it("leaves short audio alone", () => {
    expect(splitOnSilence(tone(12))).toEqual([{ start: 0, end: 12 * SR }]);
  });

  it("cuts inside the pause near the 30 s limit", () => {
    // speech 0-27 s, pause 27-28 s, speech 28-40 s
    const audio = concat(tone(27), silence(1), tone(12));
    const [a, b] = splitOnSilence(audio);
    expect(a.start).toBe(0);
    expect(a.end).toBeGreaterThanOrEqual(27 * SR);
    expect(a.end).toBeLessThanOrEqual(28 * SR);
    expect(b).toEqual({ start: a.end, end: audio.length });
  });

  it("covers the whole signal with no chunk over the limit", () => {
    const audio = concat(tone(26), silence(0.5), tone(29), silence(0.3), tone(20));
    const chunks = splitOnSilence(audio);
    expect(chunks[0].start).toBe(0);
    expect(chunks.at(-1)!.end).toBe(audio.length);
    for (let i = 1; i < chunks.length; i++) expect(chunks[i].start).toBe(chunks[i - 1].end);
    for (const c of chunks) expect(c.end - c.start).toBeLessThanOrEqual(30 * SR);
  });

  it("still splits continuous sound at the limit", () => {
    const chunks = splitOnSilence(tone(65));
    expect(chunks.length).toBe(3);
    for (const c of chunks) expect(c.end - c.start).toBeLessThanOrEqual(30 * SR);
  });
});
