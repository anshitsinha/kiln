# Results log

Measured numbers only, newest first, each with the setup that produced it.
Nothing here is a final claim — kernels are still being tuned.

## Whisper-tiny runs end to end

`test/whisper.test.ts`, JFK clip (11 s, 16 kHz), f32 weights, greedy,
NVIDIA Turing / Chromium. Transcript:

> And so my fellow Americans ask not what your country can do for you ask
> what you can do for your country.

Correct apart from punctuation — and since every kernel feeds into it,
this is the strongest correctness check in the repo.

| stage | cold (shaders compiling) | warm |
| --- | ---: | ---: |
| log-mel spectrogram (CPU) | 275 ms | 301 ms |
| encoder | 335 ms | 80 ms |
| decoder, 23 tokens + 4-token prompt | 378 ms | 282 ms (~11.7 ms/step) |
| **total** | **0.99 s** | **0.66 s (~17× real time)** |

Where the time goes now:

- **The mel spectrogram is the biggest single cost** and it's on the CPU:
  3000 FFTs in JS. Moving the STFT and mel filterbank to the GPU (two
  matmuls) is the obvious next step.
- **Decoding is latency-bound, not GPU-bound.** Each step is ~100 small
  dispatches plus a readback of all 51,865 logits to pick one argmax.
  Doing the argmax on the GPU (read back 4 bytes, not 200 KB), fusing the
  Q/K/V projections, and int4 weights are the levers.
- The encoder at 80 ms is mostly the 1500×384 matmuls and attention.

Not yet compared against anything external — that's 0.4.

## Fused attention with a decode kernel: faster than unfused everywhere

Same setup. `attention()` now picks a key-parallel kernel for ≤ 8 queries
(8 subgroups split the keys, lanes split the head dims, partial softmax
states merged at the end) and the query-parallel flash kernel otherwise.

| H×Tq×Tk | what | unfused ms | fused ms | speedup | before this change |
| --- | --- | ---: | ---: | ---: | ---: |
| 6×1500×1500 | encoder self-attention | 7.323 | 3.562 | **2.06×** | 2.00× |
| 6×100×100 | decoder self-attn, prompt | 0.090 | 0.081 | 1.11× | 1.11× |
| 6×1×448 | 1 token vs full cache | 0.126 | 0.051 | **2.47×** | 0.51× |
| 6×1×1500 | cross-attn, 1 token | 0.362 | 0.155 | **2.33×** | 0.39× |
| 6×4×1500 | cross-attn, 4 tokens | 0.365 | 0.163 | **2.24×** | 0.41× |

The decode steps went from ~2× *slower* than unfused to ~2.3× faster —
about 6× faster than the first fused kernel on the same shapes. Same
lesson as the int4 GEMV: at decode sizes the work is tiny, so what decides
speed is how much of the GPU you actually keep busy.

## Fused attention, first version: 2× on the encoder, slower for decoding

`/bench/attention.html`, Whisper-tiny shapes (6 heads, head dim 64), GPU
time, median of 7. Unfused = QKᵀ + softmax + PV (sum of the three kernels).

| H×Tq×Tk | what | unfused ms | fused ms | speedup | intermediate avoided |
| --- | --- | ---: | ---: | ---: | ---: |
| 6×1500×1500 | encoder self-attention | 7.333 | 3.661 | **2.00×** | 108 MB |
| 6×100×100 | decoder self-attn, prompt | 0.094 | 0.084 | 1.11× | 0.48 MB |
| 6×1×448 | 1 token vs full cache | 0.151 | 0.297 | **0.51×** | 0.02 MB |
| 6×1×1500 | cross-attn, 1 token | 0.376 | 0.975 | **0.39×** | 0.07 MB |
| 6×4×1500 | cross-attn, 4 tokens | 0.398 | 0.975 | 0.41× | 0.29 MB |

- With many queries (the encoder) fusing wins 2× and never writes the
  108 MB score/probability matrices.
- With one query it loses badly. The kernel parallelises over query rows,
  so a single-token step launches 6 workgroups with **one busy thread each**,
  and that thread walks all 1500 keys serially. The unfused version at least
  spreads the keys over a whole matmul.
- Fix: a decode-specific kernel that parallelises over *keys* instead
  ("flash-decoding"): split the keys across subgroups, run the online
  softmax per subgroup, merge the partial (max, sum, output) at the end.

## int4 GEMV, second attempt: 1.5–4.2× faster than f32

Same setup, `/bench/decode.html`. The fix: fewer threads per row (a knob,
4–64) with more rows per workgroup, and a `subgroupShuffleXor` butterfly
for the per-row sum (no workgroup memory, no barriers).

Lanes-per-row sweep, ms per token:

| shape (M×N×K) | what | f32 | 4 lanes | 8 | 16 | 32 | 64 | best vs f32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1×384×384 | attention projection | 0.0107 | 0.0102 | 0.0079 | **0.0075** | 0.0097 | 0.0122 | 1.4× |
| 1×1536×384 | MLP up | 0.0281 | 0.0117 | **0.0105** | 0.0141 | 0.0226 | 0.0328 | 2.7× |
| 1×384×1536 | MLP down | 0.0244 | 0.0272 | 0.0169 | **0.0142** | 0.0202 | 0.0232 | 1.7× |
| 1×51865×384 | vocab projection | 0.690 | **0.163** | 0.187 | 0.311 | 0.602 | 0.924 | **4.2×** |

- The best lane count follows **N**, not K: 16 for N = 384, 8 for 1536, 4
  for 51865. Small layers need more lanes per row just to keep the GPU busy;
  big ones have rows to spare and want fewer, busier lanes.
  `defaultLanes(N, K)` encodes that and picks the best entry in every row
  above (tuned on this one GPU).
- The vocab projection — the single biggest matmul in Whisper's decoder —
  drops from 0.69 ms to 0.16 ms per token.
- 64 lanes (the f32 GEMV's layout) is the *worst* choice for int4 at every
  shape, which is why the first attempt looked like int4 didn't help.
- Small layers (N = 384) are still mostly launch overhead at ~8 µs; the fix
  there is fusing several projections into one dispatch, not a faster GEMV.

## int4 GEMV, first attempt: not faster (yet)

Setup as below. One token, `/bench/decode.html`, median of 9.

| shape (M×N×K) | what | f32 ms | int4 ms | weights read (f32 → int4) | speedup |
| --- | --- | ---: | ---: | --- | ---: |
| 1×384×384 | attention projection | 0.0113 | 0.0110 | 0.59 → 0.09 MB | 1.02× |
| 1×1536×384 | MLP up | 0.0282 | 0.0291 | 2.36 → 0.37 MB | 0.97× |
| 1×384×1536 | MLP down | 0.0246 | 0.0216 | 2.36 → 0.37 MB | 1.14× |
| 1×51865×384 | vocab projection | 0.690 | 0.874 | 79.7 → 12.5 MB | **0.79×** |

Reading 6.4× fewer bytes bought nothing — the kernel isn't bandwidth-bound
at these sizes. With K = 384 an int4 row is only 48 u32 words, but the
kernel (copied from the f32 GEMV) gives each row 64 threads: a quarter of
them idle, the rest doing one load each, and then a full 64-lane reduction.
The time goes on scheduling and reducing, not on memory.

Next: fewer threads per row (8-16) and more rows per workgroup, reduced
with subgroup shuffles that don't touch workgroup memory.

Output error of int4 vs f32 on these random weights: worst element
0.19–0.30 of the output's RMS. Uniform random weights are close to a worst
case for symmetric 4-bit; what matters is the effect on real Whisper
transcripts (word error rate), which gets measured once the model runs.

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
