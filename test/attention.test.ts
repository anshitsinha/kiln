import { describe, expect, it } from "vitest";
import { attentionUnfused } from "../src/kernels/attention";
import { attentionRef } from "../src/reference/attention";
import { expectClose, getKiln, randomData } from "./helpers";

// [H, Tq, Tk, D, causal]
const CASES: [number, number, number, number, boolean][] = [
  [1, 1, 1, 4, false],
  [2, 5, 7, 8, false],
  [6, 100, 100, 64, true], // decoder self-attention (prompt)
  [6, 1, 64, 64, true], // decoder self-attention, one new token vs cache
  [6, 4, 1500, 64, false], // decoder cross-attention over encoder output
  [6, 1500, 1500, 64, false], // encoder self-attention
];

describe("attention (unfused)", () => {
  it.each(CASES)("H=%i Tq=%i Tk=%i D=%i causal=%s", async (H, Tq, Tk, D, causal) => {
    const kiln = await getKiln();
    const q = randomData(H * Tq * D, 1, 1);
    const k = randomData(H * Tk * D, 2, 1);
    const v = randomData(H * Tk * D, 3, 1);
    const got = await attentionUnfused(
      kiln,
      kiln.tensor(q, [H, Tq, D]),
      kiln.tensor(k, [H, Tk, D]),
      kiln.tensor(v, [H, Tk, D]),
      { causal },
    ).readF32();
    expectClose(got, attentionRef(q, k, v, H, Tq, Tk, D, { causal }), 1e-4, 1e-5);
  });

  it("rejects mismatched shapes", async () => {
    const kiln = await getKiln();
    expect(() => attentionUnfused(kiln, kiln.empty([2, 3, 8]), kiln.empty([2, 4, 16]), kiln.empty([2, 4, 16]))).toThrow(/line up/);
  });
});
