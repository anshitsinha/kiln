# Roadmap

## 0.1 — runtime + first kernels ✅

- [x] Repo, docs
- [x] Device setup with feature detection (f16, subgroups, timestamps)
- [x] Tensors on the GPU, upload/readback, pipeline cache, batched dispatch
- [x] GPU timing via timestamp queries
- [x] Test harness: kernels vs CPU references on the real GPU
- [x] Matmul: naive → tiled (shared memory) → register-blocked
- [x] GEMV for decoding (coalesced vec4 + subgroup reductions)
- [x] Matmul benchmark page (GFLOP/s, GB/s)

## 0.2 — the transformer kernel set ✅

- [x] References: erf/GELU, softmax (causal), LayerNorm, attention
- [x] Elementwise: add, mul, GELU (exact + tanh), fused bias + activation
- [x] Softmax: online max/sum, fused scale + causal mask, subgroup reduction
- [x] LayerNorm (Welford), fused residual add
- [x] int4 group-wise quantization (CPU packing + error bounds)
- [x] int4 GEMV, lanes-per-row tuned by N (1.5–4.2× over f32 for decoding)
- [x] Fused attention: query-parallel flash kernel + key-parallel decode kernel
- [x] Decode and attention benchmark pages

## 0.3 — Whisper

- [x] safetensors loader; `npm run fetch` for weights + test clip
- [x] WAV decoding, mixed-radix FFT, log-mel spectrogram (CPU)
- [x] Conv1d as im2col + matmul, fused bias + GELU
- [x] Encoder blocks
- [x] Decoder blocks with KV cache, greedy decoding
- [x] Tokenizer (decode side)
- [x] Transcribe a WAV end to end (JFK clip, 0.66 s warm for 11 s)
- [x] Scoped GPU memory (no leaks across runs)
- [ ] Mel spectrogram on the GPU
- [ ] Argmax on the GPU (don't read back 51,865 logits per token)
- [ ] int4 weights in the model + word error rate f32 vs int4 on real audio
- [ ] Demo page: load from Hugging Face, transcribe a file or the mic

## 0.4 — the comparison

- [ ] Same model, same audio: kiln vs Transformers.js (ONNX Runtime Web, WebGPU)
- [ ] Per-kernel profile, end-to-end latency, memory
- [ ] Write-up

## Later / known gaps

- [ ] Matmul tuning: vec4 loads, 8×8 per thread, double buffering
- [ ] int4 matmul for prompt/encoder (not just GEMV)
- [ ] f16 storage + math variants
- [ ] Fuse small decoder projections into fewer dispatches (launch overhead)
- [ ] Split-K across workgroups for very long caches
