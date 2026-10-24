import { it } from "vitest";
import { loadWhisper, type WhisperConfig } from "../src/model/whisper";
import { getKiln, hasAsset } from "./helpers";

const DIR = "/weights/whisper-tiny";
const ready = await hasAsset(`${DIR}/model.safetensors`);

// A profiling tool, not a test: VITE_KILN_TOOLS=1 npx vitest run test/profile-encoder.test.ts
it.runIf(ready && import.meta.env.VITE_KILN_TOOLS)("profile encoder", { timeout: 120_000 }, async () => {
  const kiln = await getKiln();
  const [weights, config] = await Promise.all([
    fetch(`${DIR}/model.safetensors`).then((r) => r.arrayBuffer()),
    fetch(`${DIR}/config.json`).then((r) => r.json() as Promise<WhisperConfig>),
  ]);
  const model = loadWhisper(kiln, weights, config);
  const mel = kiln.tensor(new Float32Array(80 * 3000), [3000, 80]);
  (await model.encode(mel)).destroy();
  const t = await kiln.profile(async () => (await model.encode(mel)).destroy());
  const by = new Map<string, number>();
  for (const k of t) {
    by.set(k.label, (by.get(k.label) ?? 0) + k.ms);
  }
  const total = t.reduce((a, k) => a + k.ms, 0);
  console.log(`total GPU ${total.toFixed(2)} ms`);
  for (const [k, v] of [...by].sort((a, b) => b[1] - a[1])) console.log(`${v.toFixed(3)} ms  ${k}`);
  const w0 = performance.now();
  for (let i = 0; i < 5; i++) (await model.encode(mel)).destroy();
  console.log(`wall ${((performance.now() - w0) / 5).toFixed(1)} ms`);
});
