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
- [x] Mel spectrogram on the GPU (301 ms → 8 ms)
- [x] Argmax on the GPU (don't read back 51,865 logits per token)
- [x] Runtime: one-round-trip readback, packed uniforms, buffer pool
- [x] int4 decoder option (same JFK transcript, decode 124 → 108 ms)
- [x] Demo page: load from Hugging Face, transcribe a file, the mic, or a sample
- [x] Word error rate f32 vs int4 on LibriSpeech (11.91% / 14.35%)
- [x] Long audio: split at pauses, chunk by chunk (65 s → 8.6% WER)
- [ ] int4 for the embedding lookup and encoder (memory: int4 model is still 136 MB)

## 0.4 — the comparison

- [x] Same model, same audio: kiln vs Transformers.js — 3.6× faster, same WER
- [ ] More GPUs / browsers (integrated graphics, Apple Silicon, Safari)
- [ ] Bigger models (whisper-base / small), where kernels matter more than overhead
- [ ] Asymmetric int4 (per-group zero points) to close the q4 accuracy gap
- [ ] Write-up

## Later / known gaps

- [ ] Matmul tuning: vec4 loads, 8×8 per thread, double buffering
- [ ] int4 matmul for prompt/encoder (not just GEMV)
- [ ] f16 storage + math variants
- [ ] Fuse small decoder projections into fewer dispatches (launch overhead)
- [ ] Split-K across workgroups for very long caches
