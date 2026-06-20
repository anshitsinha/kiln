# Roadmap

## 0.1 — runtime + first kernels

- [x] Repo, docs
- [x] Device setup with feature detection (f16, subgroups, timestamps)
- [x] Tensors on the GPU, upload/readback, pipeline cache, batched dispatch
- [x] GPU timing via timestamp queries
- [x] Test harness: kernels vs CPU references on the real GPU
- [x] Matmul: naive → tiled (shared memory) → register-blocked
- [x] GEMV for decoding (coalesced vec4 + subgroup reductions)
- [x] Matmul benchmark page (GFLOP/s, GB/s)
- [ ] Matmul tuning: vec4 loads, 8×8 per thread, double buffering

## 0.2 — the transformer kernel set

- [ ] Elementwise: add, GELU, fused bias+GELU
- [ ] Softmax (subgroup reductions where available)
- [ ] LayerNorm
- [ ] f16 storage variants
- [ ] int4 weight-only quantized matmul (dequantize inside the kernel)
- [ ] Fused attention (FlashAttention-style online softmax, tiled K/V)

## 0.3 — Whisper

- [ ] safetensors loader (streamed from Hugging Face)
- [ ] Log-mel spectrogram
- [ ] Encoder (conv1d front end + transformer blocks)
- [ ] Decoder with KV cache, greedy decoding
- [ ] Tokenizer (decode side)
- [ ] Transcribe a WAV end to end

## 0.4 — the comparison

- [ ] Same model, same audio: kiln vs Transformers.js (ONNX Runtime Web, WebGPU)
- [ ] Per-kernel profile, end-to-end latency, memory
- [ ] Write-up
