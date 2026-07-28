import { expect } from "vitest";
import { Kiln } from "../src";

let shared: Promise<Kiln> | null = null;

/** One device for the whole test run - creating devices is slow. */
export function getKiln(): Promise<Kiln> {
  shared ??= Kiln.create();
  return shared;
}

/**
 * Whether a downloaded asset (weights/, from `npm run fetch`) is available.
 * Tests that need one use it.runIf(...) so they show up as skipped rather
 * than quietly passing.
 */
export async function hasAsset(path: string): Promise<boolean> {
  try {
    const res = await fetch(path, { method: "HEAD" });
    // vite's dev server falls back to index.html for unknown paths
    return res.ok && !(res.headers.get("content-type") ?? "").includes("text/html");
  } catch {
    return false;
  }
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
