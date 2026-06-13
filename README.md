# kiln

A from-scratch WebGPU inference engine for transformer models, written around
what WebGPU actually is — not a port of a native runtime.

> **Status: early.** Building up the kernel library and runtime. No model runs
> end to end yet; see the [roadmap](docs/ROADMAP.md). Performance claims in
> this repo are only ever measured numbers, with the setup that produced them.

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
browser (first target: **Whisper** for speech-to-text).

## Design rules

1. **Every kernel has a CPU reference** and a test that compares them on the
   real GPU. A fast kernel that's wrong is worthless.
2. **Measure, don't guess.** Kernels are timed with GPU timestamp queries, and
   the benchmark page reports the hardware it ran on.
3. **Fuse where the memory traffic is.** Inference in the browser is mostly
   bandwidth-bound, so the wins come from reading weights fewer times.

## Getting started

Requires Node 20+ and a browser with WebGPU (Chrome/Edge 113+, recent Safari).

```bash
npm install
npm run dev     # kernel playground + benchmarks at http://localhost:5173
npm test        # kernel tests, run on your real GPU in a headless browser
```

Tests use Playwright with a locally installed Chromium-based browser. Set
`KILN_BROWSER` to its executable path if it isn't found automatically.

## Layout

```
kiln/
├── src/
│   ├── gpu/         # device, tensors, pipeline cache, GPU timing
│   ├── kernels/     # WGSL kernels + their dispatch code
│   └── reference/   # plain-JS reference implementations (the spec)
├── test/            # GPU-vs-reference tests (vitest browser mode)
├── bench/           # benchmark pages
└── docs/
```

## License

[MIT](LICENSE)
