import { matmulRef } from "./matmul";
import { softmaxRef } from "./ops";

/**
 * Scaled dot-product attention, per head:
 *
 *   O = softmax(Q K^T * scale [+ causal mask]) V
 *
 *   Q: [H, Tq, D]   K, V: [H, Tk, D]   O: [H, Tq, D]
 *
 * Causal masking uses the same offset rule as softmaxRef: query row i may
 * attend to keys 0 .. Tk - Tq + i, so with a KV cache (Tq < Tk) the newest
 * query sees the whole cache.
 */
export function attentionRef(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  H: number,
  Tq: number,
  Tk: number,
  D: number,
  opts: { scale?: number; causal?: boolean } = {},
): Float32Array {
  const scale = opts.scale ?? 1 / Math.sqrt(D);
  const out = new Float32Array(H * Tq * D);
  for (let h = 0; h < H; h++) {
    const qh = q.subarray(h * Tq * D, (h + 1) * Tq * D);
    const kh = k.subarray(h * Tk * D, (h + 1) * Tk * D);
    const vh = v.subarray(h * Tk * D, (h + 1) * Tk * D);
    const scores = matmulRef(qh, kh, Tq, Tk, D, { transposeB: true }).map((s) => s * scale);
    const p = softmaxRef(scores, Tq, Tk, opts.causal);
    out.set(matmulRef(p, vh, Tq, D, Tk), h * Tq * D);
  }
  return out;
}
