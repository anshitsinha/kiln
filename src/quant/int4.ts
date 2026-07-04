/**
 * Group-wise symmetric int4 weight quantization (CPU side).
 *
 * A Linear weight W is [N, K] (PyTorch layout: one row per output). Each row
 * is cut into groups of GROUP consecutive weights along K; each group gets
 * one f32 scale s = max|w| / 7 and every weight becomes
 *
 *   q = clamp(round(w / s), -8, 7)        stored as the nibble q + 8 (0..15)
 *
 * Eight nibbles pack into one u32, low nibble first, so word j of a row
 * holds k = 8j .. 8j+7.
 *
 * Cost per weight: 4 bits + 32 bits / GROUP = 0.625 bytes, vs 4 for f32.
 * Decoding a token streams every weight once, so that's ~6.4x less memory
 * traffic on the part of inference that's bandwidth-bound.
 *
 * Symmetric because it keeps the kernel to one multiply per group and the
 * zero point a constant 8. Asymmetric (per-group min/max) is a little more
 * accurate on skewed groups and is an easy follow-up if the error numbers
 * say it's needed.
 */

export const GROUP = 32;

export interface Int4Weights {
  /** [N, K/8] packed nibbles */
  packed: Uint32Array;
  /** [N, K/GROUP] per-group scales */
  scales: Float32Array;
  N: number;
  K: number;
}

export function quantizeInt4(w: ArrayLike<number>, N: number, K: number): Int4Weights {
  if (K % GROUP !== 0) throw new Error(`quantizeInt4: K (${K}) must be a multiple of ${GROUP}`);
  if (w.length !== N * K) throw new Error(`quantizeInt4: expected ${N * K} weights, got ${w.length}`);
  const groups = K / GROUP;
  const packed = new Uint32Array((N * K) / 8);
  const scales = new Float32Array(N * groups);

  for (let n = 0; n < N; n++) {
    for (let g = 0; g < groups; g++) {
      const base = n * K + g * GROUP;
      let amax = 0;
      for (let i = 0; i < GROUP; i++) amax = Math.max(amax, Math.abs(w[base + i]));
      // an all-zero group would give s = 0 and 0/0 below; any scale works
      const s = amax > 0 ? amax / 7 : 1;
      scales[n * groups + g] = s;
      for (let i = 0; i < GROUP; i++) {
        const q = Math.max(-8, Math.min(7, Math.round(w[base + i] / s)));
        const k = g * GROUP + i;
        const word = (n * K + k) >>> 3;
        packed[word] |= (q + 8) << ((k & 7) * 4);
      }
    }
  }
  return { packed, scales, N, K };
}

/** Back to f32, exactly as the GPU kernel interprets the data. */
export function dequantizeInt4({ packed, scales, N, K }: Int4Weights): Float32Array {
  const groups = K / GROUP;
  const out = new Float32Array(N * K);
  for (let n = 0; n < N; n++) {
    for (let k = 0; k < K; k++) {
      const nib = (packed[(n * K + k) >>> 3] >>> ((k & 7) * 4)) & 0xf;
      out[n * K + k] = (nib - 8) * scales[n * groups + ((k / GROUP) | 0)];
    }
  }
  return out;
}

/** Bytes needed on the GPU, for reporting. */
export function int4Bytes(N: number, K: number): number {
  return N * (K / 2) + N * (K / GROUP) * 4;
}
