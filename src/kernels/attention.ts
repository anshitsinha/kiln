import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { matmul } from "./matmul";
import { softmax } from "./softmax";

/**
 * Scaled dot-product attention.
 *
 *   q: [H, Tq, D]   k, v: [H, Tk, D]   ->   [H, Tq, D]
 */

export interface AttentionOptions {
  scale?: number;
  causal?: boolean;
}

function dims(q: Tensor, k: Tensor, v: Tensor) {
  if (q.shape.length !== 3 || k.shape.length !== 3 || v.shape.length !== 3) {
    throw new Error("attention: q, k, v must be [heads, tokens, headDim]");
  }
  const [H, Tq, D] = q.shape;
  const [Hk, Tk, Dk] = k.shape;
  if (Hk !== H || Dk !== D || v.shape[0] !== H || v.shape[1] !== Tk || v.shape[2] !== D) {
    throw new Error(`attention: shapes don't line up - q ${q.shape}, k ${k.shape}, v ${v.shape}`);
  }
  return { H, Tq, Tk, D };
}

/**
 * The unfused baseline: three kernels, the way a general graph runtime runs
 * it. QK^T writes the full [H, Tq, Tk] score matrix to memory, softmax reads
 * and rewrites it, and PV reads it again - for Whisper's encoder (6 heads,
 * 1500 tokens) that's a 54 MB intermediate, round-tripped three times.
 */
export function attentionUnfused(kiln: Kiln, q: Tensor, k: Tensor, v: Tensor, opts: AttentionOptions = {}): Tensor {
  const { D } = dims(q, k, v);
  const scores = matmul(kiln, q, k, { transposeB: true, variant: "blocked" });
  const probs = softmax(kiln, scores, { scale: opts.scale ?? 1 / Math.sqrt(D), causal: opts.causal });
  return matmul(kiln, probs, v, { variant: "blocked" });
}

export { dims as attentionDims };
