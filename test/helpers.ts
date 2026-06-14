import { expect } from "vitest";
import { Kiln } from "../src";

let shared: Promise<Kiln> | null = null;

/** One device for the whole test run - creating devices is slow. */
export function getKiln(): Promise<Kiln> {
  shared ??= Kiln.create();
  return shared;
}

/** Deterministic pseudo-random data in [-scale, scale). */
export function randomData(n: number, seed = 1, scale = 1): Float32Array {
  let s = seed >>> 0 || 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    out[i] = ((s >>> 0) / 4294967296) * 2 * scale - scale;
  }
  return out;
}

/**
 * Compare GPU output to a reference with a mixed tolerance:
 * |gpu - ref| <= atol + rtol * |ref|. Reports the worst element on failure,
 * which is far more useful than "arrays differ".
 */
export function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, rtol = 1e-5, atol = 1e-5) {
  expect(actual.length).toBe(expected.length);
  let worst = -1;
  let worstExcess = 0;
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const excess = Math.abs(a - e) - (atol + rtol * Math.abs(e));
    if (Number.isNaN(a) !== Number.isNaN(e) || excess > worstExcess) {
      worst = i;
      worstExcess = Number.isNaN(excess) ? Infinity : excess;
    }
  }
  if (worst >= 0) {
    throw new Error(
      `mismatch at index ${worst}: got ${actual[worst]}, expected ${expected[worst]} ` +
        `(rtol ${rtol}, atol ${atol})`,
    );
  }
}
