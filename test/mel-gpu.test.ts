import { describe, expect, it } from "vitest";
import { logMelSpectrogram, N_FRAMES, N_MELS, SAMPLE_RATE } from "../src/audio/mel";
import { logMelSpectrogramGpu } from "../src/audio/mel-gpu";
import { decodeWav } from "../src/audio/wav";
import { getKiln, hasAsset, randomData } from "./helpers";

const JFK = "/weights/samples/jfk.wav";
const hasJfk = await hasAsset(JFK);

/** GPU output is [frames, mels]; the CPU one is [mels, frames]. */
async function compare(samples: Float32Array) {
  const kiln = await getKiln();
  const gpu = await kiln.scope(async () => logMelSpectrogramGpu(kiln, samples).readF32());
  const cpu = logMelSpectrogram(samples);
  let worst = 0;
  let sum = 0;
  for (let t = 0; t < N_FRAMES; t++) {
    for (let m = 0; m < N_MELS; m++) {
      const d = Math.abs(gpu[t * N_MELS + m] - cpu[m * N_FRAMES + t]);
      worst = Math.max(worst, d);
      sum += d;
    }
  }
  return { worst, mean: sum / (N_FRAMES * N_MELS) };
}

describe("GPU log-mel spectrogram", () => {
  // f32 DFT-as-matmul vs the f64 FFT path. The output is in log10 units / 4,
  // so 1e-3 here is a ~1% error in power on the quietest bins that survive
  // the max-8 floor - far below anything the model can notice.
  it("matches the CPU version on noise", async () => {
    const { worst, mean } = await compare(randomData(SAMPLE_RATE * 5, 3, 0.5));
    expect(worst).toBeLessThan(1e-3);
    expect(mean).toBeLessThan(1e-5);
  });

  it("matches on a tone (sharp spectral peak)", async () => {
    const tone = Float32Array.from({ length: SAMPLE_RATE * 2 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / SAMPLE_RATE));
    const { worst } = await compare(tone);
    expect(worst).toBeLessThan(1e-3);
  });

  it("is exactly -1.5 for silence", async () => {
    const kiln = await getKiln();
    const out = await kiln.scope(async () => logMelSpectrogramGpu(kiln, new Float32Array(1000)).readF32());
    expect(out.every((v) => v === -1.5)).toBe(true);
  });

  it.runIf(hasJfk)("matches on real speech", async () => {
    const { samples } = decodeWav(await (await fetch(JFK)).arrayBuffer());
    const { worst, mean } = await compare(samples);
    console.log(`jfk: worst ${worst.toExponential(2)}, mean ${mean.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-3);
  });
});
