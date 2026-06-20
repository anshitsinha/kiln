# Results log

Measured numbers only, newest first, each with the setup that produced it.
Nothing here is a final claim — kernels are still being tuned.

## GEMV for decoding (f32)

Same setup as below. With one token, the useful metric is how fast the
weights stream from memory (weights GB/s = N·K·4 bytes / time).

| shape (M×N×K) | what | naive | tiled | blocked | **gemv** |
| --- | --- | ---: | ---: | ---: | ---: |
| 1×1536×384 | decoder MLP, 1 token | 0.089 ms · 27 GB/s | 0.063 ms · 37 GB/s | 0.111 ms · 21 GB/s | **0.021 ms · 113 GB/s** |
| 1×51865×384 | vocab projection, 1 token | 1.575 ms · 51 GB/s | 1.882 ms · 42 GB/s | 2.85 ms · 28 GB/s | **0.500 ms · 159 GB/s** |

- **5.3–5.7× faster than the blocked matmul** at M = 1, by giving each
  output 64 threads that read the weight row as coalesced vec4s and
  combining partial sums with `subgroupAdd`.
- 159 GB/s is roughly half of what this class of GPU's memory can deliver,
  and everything here is f32. Storing weights as int4 cuts the bytes per
  weight from 4 to ~0.56 (with scales), which is where the next big
  decode-speed win is.

## Matmul variants (f32)

Setup: NVIDIA Turing (Brave / Chromium 157, headless, `--enable-unsafe-webgpu`),
Windows 11. GPU time from timestamp queries, median of 7 runs after warm-up.
Reproduce at `/bench/matmul.html`.

| shape (M×N×K) | what | naive | tiled | blocked |
| --- | --- | ---: | ---: | ---: |
| 1024×1024×1024 | square | 103 GFLOP/s | 417 | **998** |
| 2048×2048×2048 | square | 113 | 432 | **1114** |
| 1500×1536×384 | Whisper-tiny encoder MLP up | 51 | 331 | **854** |
| 1500×384×1536 | Whisper-tiny encoder MLP down | 52 | 330 | **810** |
| 1500×1152×384 | Whisper-tiny fused QKV | 51 | 330 | **826** |
| 1×1536×384 | decoder step, 1 token | 14 | **19** | 11 |

### What this says

- **Register blocking is worth ~10× over naive** and ~2.5× over shared-memory
  tiling on large shapes. Each thread keeps a 4×4 block of outputs in
  registers, so every value fetched from workgroup memory feeds 4 FMAs.
- **~1.1 TFLOP/s is still well short of this GPU's f32 peak.** Obvious next
  steps: vec4 loads, 8×8 outputs per thread, double-buffered tiles, f16.
- **One-token decoding is a different problem.** With M = 1 the 64×64
  blocked tile wastes 63 of its 64 rows and comes out *slowest*. A decoder
  step is a matrix-vector product, bound by how fast the weights can be
  streamed from memory, not by FLOPs. It needs its own kernel (GEMV with
  coalesced weight reads and subgroup reductions) — and that's exactly where
  int4 weights will pay off, since they cut the bytes streamed by ~8×.
