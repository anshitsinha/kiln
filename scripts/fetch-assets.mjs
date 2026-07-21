// Downloads the model files and test audio into weights/ (gitignored).
// Usage: npm run fetch [-- --model whisper-tiny]
//
// Skips files that are already there with the right size, so it's cheap to
// re-run.
import { createWriteStream, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const modelArg = process.argv.indexOf("--model");
const model = modelArg > 0 ? process.argv[modelArg + 1] : "whisper-tiny";
const hf = (file) => `https://huggingface.co/openai/${model}/resolve/main/${file}`;

const assets = [
  { url: hf("model.safetensors"), to: `${model}/model.safetensors` },
  { url: hf("config.json"), to: `${model}/config.json` },
  { url: hf("tokenizer.json"), to: `${model}/tokenizer.json` },
  // JFK's "ask not" line - public domain, and the standard whisper.cpp sample
  { url: "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav", to: "samples/jfk.wav" },
];

for (const { url, to } of assets) {
  const dest = join(root, "weights", to);
  mkdirSync(dirname(dest), { recursive: true });

  const head = await fetch(url, { method: "HEAD", redirect: "follow" });
  if (!head.ok) throw new Error(`${url}: HTTP ${head.status}`);
  const size = Number(head.headers.get("content-length") ?? 0);
  if (existsSync(dest) && size && statSync(dest).size === size) {
    console.log(`ok       ${to}`);
    continue;
  }

  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  process.stdout.write(`fetching ${to} (${(size / 1e6).toFixed(1)} MB)... `);
  await pipeline(Readable.fromWeb(res.body), createWriteStream(dest));
  console.log("done");
}
