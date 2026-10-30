// Downloads the model files and test audio into weights/ (gitignored).
// Usage: npm run fetch [-- --model whisper-tiny]
//
// Skips files that are already there with the right size, so it's cheap to
// re-run.
import { setDefaultResultOrder } from "node:dns";
import { createWriteStream, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

// Hugging Face's CDN resolves to IPv6 too; on networks where v6 is routed
// but broken, Node's fetch hangs until a connect timeout instead of falling
// back. Asking for v4 first avoids that.
setDefaultResultOrder("ipv4first");

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

// --- eval set ---------------------------------------------------------------------
// The 73-utterance LibriSpeech validation sample Hugging Face keeps for tests
// (hf-internal-testing/librispeech_asr_dummy), as FLAC + reference text.
// Fetched through the datasets-server API, whose audio URLs are signed and
// short-lived, so they're downloaded straight away.

const evalDir = join(root, "weights", "eval", "librispeech-dummy");
const manifestPath = join(evalDir, "manifest.json");
if (existsSync(manifestPath)) {
  console.log("ok       eval/librispeech-dummy");
} else {
  mkdirSync(evalDir, { recursive: true });
  const api = "https://datasets-server.huggingface.co/rows?dataset=hf-internal-testing/librispeech_asr_dummy&config=clean&split=validation";
  // small pages: asking for 100 rows at once makes the API time out (502)
  // generating that many signed audio URLs
  const PAGE = 10;
  const getPage = async (offset) => {
    for (let attempt = 1; ; attempt++) {
      const res = await fetch(`${api}&offset=${offset}&length=${PAGE}`);
      if (res.ok) return res.json();
      if (attempt === 4) throw new Error(`datasets-server offset ${offset}: HTTP ${res.status}`);
      await new Promise((r) => setTimeout(r, 1000 * attempt));
    }
  };
  const manifest = [];
  for (let offset = 0; ; offset += PAGE) {
    const page = await getPage(offset);
    for (const { row_idx, row } of page.rows) {
      const file = `${String(row_idx).padStart(3, "0")}.flac`;
      const audio = await fetch(row.audio[0]?.src ?? row.audio.src);
      if (!audio.ok || !audio.body) throw new Error(`eval audio ${row_idx}: HTTP ${audio.status}`);
      await pipeline(Readable.fromWeb(audio.body), createWriteStream(join(evalDir, file)));
      manifest.push({ id: row.id ?? String(row_idx), file, text: row.text });
    }
    process.stdout.write(`\rfetching eval/librispeech-dummy: ${manifest.length}/${page.num_rows_total}`);
    if (offset + PAGE >= page.num_rows_total) break;
  }
  // written last, so an interrupted download gets retried next time
  const { writeFileSync } = await import("node:fs");
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 1));
  console.log(" done");
}
