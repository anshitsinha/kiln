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

- [ ] safetensors loader (streamed from Hugging Face)
- [ ] Log-mel spectrogram
- [ ] Conv1d (+ GELU) for the encoder front end
- [ ] Encoder blocks
- [ ] Decoder blocks with KV cache, greedy decoding
- [ ] Tokenizer (decode side)
- [ ] Transcribe a WAV end to end
- [ ] Word error rate f32 vs int4 on real audio

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
