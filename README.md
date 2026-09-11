# kiln

A from-scratch WebGPU inference engine for transformer models, written around
what WebGPU actually is — not a port of a native runtime.

**It runs OpenAI's Whisper (speech-to-text) entirely in the browser**, on
hand-written WGSL kernels: no ONNX, no WASM runtime, no server.
[Try the demo](https://kiln-brown-seven.vercel.app/demo.html) (needs a
WebGPU browser, e.g. desktop Chrome or Edge).

> Performance claims in this repo are only ever measured numbers, with the
> setup that produced them — see [docs/RESULTS.md](docs/RESULTS.md).

## vs Transformers.js

Whisper-tiny on 73 LibriSpeech utterances (481 s of audio), same browser
and GPU (NVIDIA Turing, Chromium), same scoring, warm:

| engine | WER | time | speed |
| --- | ---: | ---: | ---: |
| **kiln** (f32) | 11.91% | **11.3 s** | **43× real time** |
| Transformers.js 4.3.2 (f32, WebGPU) | 11.91% | 40.2 s | 12× real time |

**3.6× faster with identical accuracy** (identical WER and transcripts).
One GPU, one model size, Transformers.js at its defaults — reproduce it at
`/bench/compare.html` after `npm run fetch`.

## Where it's at

Whisper-tiny, 11 s clip, NVIDIA Turing / Chromium, warm:

| | spectrogram | encoder | decoder (23 tokens) | total | speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| f32 | 8 ms | 65 ms | 124 ms | ~0.20 s | ~55× real time |
| int4 decoder | 8 ms | 51 ms | 108 ms | ~0.17 s | ~66× real time |

Kernel-level results so far (all vs kiln's own simpler kernels):

- **Matmul:** register-blocked kernel ~1.1 TFLOP/s, ~10× naive.
- **Decoding:** dedicated GEMV 5.7× faster than the blocked matmul at one
  token; int4 weights another 1.5–4.2× on top.
- **Attention:** fused FlashAttention-style kernel 2.1× over the unfused
  version on the encoder; a key-parallel decode kernel 2.2–2.5× for decoding.
- **Runtime:** batched submission, single-round-trip readback, packed
  uniforms and a buffer pool took a decode step from 10.5 ms to 5.1 ms with
  the GPU work unchanged — on WebGPU, small-batch inference is dominated by
  per-dispatch overhead.

## Why

Running models in the browser means zero server GPU cost and data that never
leaves the device. Today that mostly goes through ONNX Runtime Web or
Transformers.js, which compile general-purpose graphs to WebGPU. kiln goes the
other way: a small set of hand-written WGSL kernels designed around WebGPU's
specific constraints —

- no f64 and optional f16,
- 16–32KB of workgroup memory,
- per-dispatch overhead that punishes many tiny kernels,
- subgroup operations that may or may not exist,
- buffer binding limits,

— and fused aggressively for the models people actually want to run in a
browser.

## Design rules

1. **Every kernel has a CPU reference** and a test that compares them on the
   real GPU. A fast kernel that's wrong is worthless. The end-to-end test is
   a real transcript: if any kernel is off, the text turns to garbage.
2. **Measure, don't guess.** Kernels are timed with GPU timestamp queries, and
   the benchmark pages report the hardware they ran on.
3. **Fuse where the memory traffic is**, and **cut dispatches where the
   overhead is**.

## Getting started

Requires Node 20+ and a browser with WebGPU (Chrome/Edge 113+, recent Safari).

```bash
npm install
npm run fetch   # whisper-tiny weights + a test clip into weights/ (~150 MB)
npm run dev     # demo + benchmarks at http://localhost:5173
npm test        # ~230 tests, run on your real GPU in a headless browser
```

Tests use Playwright with a locally installed Chromium-based browser. Set
`KILN_BROWSER` to its executable path if it isn't found automatically.
Tests that need the downloaded weights are skipped (visibly) without them.

## Using it

```ts
import { Kiln } from "./src";
import { decodeAudio } from "./src/audio/decode";
import { Tokenizer } from "./src/model/tokenizer";
import { transcribe } from "./src/model/transcribe";
import { loadWhisper } from "./src/model/whisper";

const kiln = await Kiln.create();
const model = loadWhisper(kiln, weightsArrayBuffer, config, { quantize: "int4" });
const samples = await decodeAudio(await file.arrayBuffer());
const { text } = await transcribe(model, new Tokenizer(tokenizerJson), samples);
```

## Layout

```
kiln/
├── src/
│   ├── gpu/         # device, tensors, batched dispatch, buffer pool, GPU timing
│   ├── kernels/     # WGSL kernels: matmul, gemv (f32 + int4), attention, softmax,
│   │                #   layernorm, elementwise, argmax, im2col, head layouts
│   ├── quant/       # int4 group-wise quantization
│   ├── audio/       # WAV/Web Audio decoding, FFT, log-mel (CPU + GPU)
│   ├── model/       # safetensors, tokenizer, Whisper, transcription, Hub fetching
│   └── reference/   # plain-JS reference implementations (the spec)
├── test/            # GPU-vs-reference tests (vitest browser mode)
├── bench/           # benchmark pages
├── demo/            # the speech-to-text demo
└── docs/            # roadmap, results log
```

## License

[MIT](LICENSE)
