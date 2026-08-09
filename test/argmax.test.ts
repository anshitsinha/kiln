import { describe, expect, it } from "vitest";
import { argmax } from "../src/kernels/argmax";
import { getKiln, randomData } from "./helpers";

function argmaxRef(x: ArrayLike<number>, from: number, to: number, extra?: number): number {
  let best = -1;
  for (let i = from; i < to; i++) if (best < 0 || x[i] > x[best]) best = i;
  if (extra !== undefined && x[extra] > x[best]) best = extra;
  return best;
}

describe("argmax", () => {
  it.each([1, 7, 300, 51865])("finds the max of a %i-wide row", async (cols) => {
    const kiln = await getKiln();
    const x = randomData(cols, cols, 10);
    const [got] = await argmax(kiln, kiln.tensor(x, [1, cols])).read();
    expect(got).toBe(argmaxRef(x, 0, cols));
  });

  it("handles several rows", async () => {
    const kiln = await getKiln();
    const [rows, cols] = [4, 1000];
    const x = randomData(rows * cols, 3, 5);
    const got = Array.from(await argmax(kiln, kiln.tensor(x, [rows, cols])).read());
    expect(got).toEqual([0, 1, 2, 3].map((r) => argmaxRef(x, r * cols, (r + 1) * cols) - r * cols));
  });

  it("breaks ties toward the lower index", async () => {
    const kiln = await getKiln();
    const x = new Float32Array(1000).fill(1);
    const [got] = await argmax(kiln, kiln.tensor(x, [1, 1000])).read();
    expect(got).toBe(0);
  });

  it("restricts to a prefix plus one extra column (text tokens + end-of-text)", async () => {
    const kiln = await getKiln();
    const x = randomData(100, 5, 1);
    x[90] = 50; // biggest overall, but outside the allowed range
    x[95] = 20; // the allowed extra column
    const t = kiln.tensor(x, [1, 100]);
    expect((await argmax(kiln, t, { limit: 80 }).read())[0]).toBe(argmaxRef(x, 0, 80));
    expect((await argmax(kiln, t, { limit: 80, extra: 95 }).read())[0]).toBe(95);
  });
});
