import { f16BitsToF32, f32ToF16Bits } from "../gpu/tensor";

/**
 * Group-wise int4 weight quantization (CPU side).
 *
 * A Linear weight W is [N, K] (PyTorch layout: one row per output). Each row
 * is cut into groups of GROUP consecutive weights along K. Each group gets a
 * scale s and an offset m, and every weight becomes a nibble q in 0..15 with
 *
 *   w ~= q * s + m
 *
 * Two ways to pick s and m:
 *
 *  - "asymmetric" (default): m = the group's min, s = (max - min) / 15. All
 *    16 levels span exactly the values the group has. Weight groups are
 *    rarely centred on zero, and a skewed group wastes half of a symmetric
 *    grid on values it never takes.
 *  - "symmetric": s = max|w| / 7, m = -8s, i.e. q - 8 in -8..7 times s. Kept
 *    for comparison (it's what kiln shipped first).
 *
 * s and m are stored as two f16s packed in one u32 per group (s in the low
 * half), so the offset costs no extra memory over a lone f32 scale. f16's
 * relative error (~5e-4) is noise next to a 1/15 quantization step.
 *
 * Eight nibbles pack into one u32, low nibble first, so word j of a row
 * holds k = 8j .. 8j+7.
 *
 * Cost per weight: 4 bits + 32 bits / GROUP = 0.625 bytes, vs 4 for f32.
 * Decoding a token streams every weight once, so that's ~6.4x less memory
 * traffic on the part of inference that's bandwidth-bound.
 */

export const GROUP = 32;

export type Int4Mode = "asymmetric" | "symmetric";

export interface Int4Weights {
  /** [N, K/8] packed nibbles */
  packed: Uint32Array;
  /** [N, K/GROUP] per-group (scale, offset) as two f16s, scale in the low half */
  params: Uint32Array;
  N: number;
  K: number;
}

const f16 = (x: number) => f16BitsToF32(f32ToF16Bits(x));

export function quantizeInt4(w: ArrayLike<number>, N: number, K: number, mode: Int4Mode = "asymmetric"): Int4Weights {
  if (K % GROUP !== 0) throw new Error(`quantizeInt4: K (${K}) must be a multiple of ${GROUP}`);
  if (w.length !== N * K) throw new Error(`quantizeInt4: expected ${N * K} weights, got ${w.length}`);
  const groups = K / GROUP;
  const packed = new Uint32Array((N * K) / 8);
  const params = new Uint32Array(N * groups);

  for (let n = 0; n < N; n++) {
    for (let g = 0; g < groups; g++) {
      const base = n * K + g * GROUP;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < GROUP; i++) {
        lo = Math.min(lo, w[base + i]);
        hi = Math.max(hi, w[base + i]);
      }
      let s: number;
      let m: number;
      if (mode === "symmetric") {
        const amax = Math.max(-lo, hi);
        // an all-zero group would give s = 0 and 0/0 below; any scale works
        s = f16(amax > 0 ? amax / 7 : 1);
        m = f16(-8 * s);
      } else {
        // a constant group: any scale works, the offset carries the value
        s = f16(hi > lo ? (hi - lo) / 15 : 1);
        m = f16(lo);
      }
      params[n * groups + g] = f32ToF16Bits(s) | (f32ToF16Bits(m) << 16);
      // quantize against the f16-rounded s and m - the values the GPU uses
      for (let i = 0; i < GROUP; i++) {
        const q = Math.max(0, Math.min(15, Math.round((w[base + i] - m) / s)));
        const k = g * GROUP + i;
        const word = (n * K + k) >>> 3;
        packed[word] |= q << ((k & 7) * 4);
      }
    }
  }
  return { packed, params, N, K };
}

/** A group's (scale, offset), unpacked. */
export function groupParams(params: Uint32Array, i: number): [scale: number, offset: number] {
  return [f16BitsToF32(params[i] & 0xffff), f16BitsToF32(params[i] >>> 16)];
}

/** Back to f32, exactly as the GPU kernel interprets the data. */
export function dequantizeInt4({ packed, params, N, K }: Int4Weights): Float32Array {
  const groups = K / GROUP;
  const out = new Float32Array(N * K);
  for (let n = 0; n < N; n++) {
    for (let k = 0; k < K; k++) {
      const nib = (packed[(n * K + k) >>> 3] >>> ((k & 7) * 4)) & 0xf;
      const [s, m] = groupParams(params, n * groups + ((k / GROUP) | 0));
      out[n * K + k] = nib * s + m;
    }
  }
  return out;
}

/** Bytes needed on the GPU, for reporting. */
export function int4Bytes(N: number, K: number): number {
  return N * (K / 2) + N * (K / GROUP) * 4;
}
