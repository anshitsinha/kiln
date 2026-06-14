import type { Kiln } from "./kiln";

export type DType = "f32" | "f16" | "u32";

export const BYTES: Record<DType, number> = { f32: 4, f16: 2, u32: 4 };

export function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * A dense, row-major array living in a GPU storage buffer.
 *
 * Deliberately minimal: no strides, no views, no autograd. Kernels take
 * tensors and produce tensors; anything fancier (transposes, slicing) is a
 * kernel too, so the cost is visible.
 */
export class Tensor {
  readonly size: number;

  constructor(
    readonly kiln: Kiln,
    readonly buffer: GPUBuffer,
    readonly shape: readonly number[],
    readonly dtype: DType,
  ) {
    this.size = numel(shape);
  }

  get byteLength(): number {
    return this.size * BYTES[this.dtype];
  }

  /** Copy back to the CPU. f16 tensors come back widened to f32. */
  async read(): Promise<Float32Array | Uint32Array> {
    const bytes = await this.kiln.readBuffer(this.buffer, this.byteLength);
    if (this.dtype === "u32") return new Uint32Array(bytes);
    if (this.dtype === "f32") return new Float32Array(bytes);
    return f16ToF32(new Uint16Array(bytes));
  }

  async readF32(): Promise<Float32Array> {
    const data = await this.read();
    if (data instanceof Float32Array) return data;
    throw new Error(`readF32 on a ${this.dtype} tensor`);
  }

  destroy(): void {
    this.buffer.destroy();
  }

  toString(): string {
    return `Tensor<${this.dtype}>[${this.shape.join("x")}]`;
  }
}

// --- f16 <-> f32 (CPU side, for uploads and readback) ----------------------

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/** Round-to-nearest-even f32 -> f16 bits. Handles subnormals, inf and NaN. */
export function f32ToF16Bits(value: number): number {
  f32Scratch[0] = value;
  const x = u32Scratch[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;

  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf / NaN
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // overflow -> inf
  if (e <= 0) {
    // subnormal (or zero) in f16
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const mid = 1 << (shift - 1);
    if (rem > mid || (rem === mid && half & 1)) half++;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && half & 1)) half++; // may carry into exp; that's correct
  return sign | half;
}

export function f16BitsToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

export function f32ToF16(data: ArrayLike<number>): Uint16Array {
  const out = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = f32ToF16Bits(data[i]);
  return out;
}

export function f16ToF32(data: Uint16Array): Float32Array {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = f16BitsToF32(data[i]);
  return out;
}
