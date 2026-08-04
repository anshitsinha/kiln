import { describe, expect, it } from "vitest";
import { attention, attentionUnfused } from "../src/kernels/attention";
import { attentionRef } from "../src/reference/attention";
import { expectClose, getKiln, randomData } from "./helpers";

// [H, Tq, Tk, D, causal]
const CASES: [number, number, number, number, boolean][] = [
  [1, 1, 1, 4, false],
  [2, 5, 7, 8, false],
  [1, 70, 70, 16, true], // causal across more than one query block and key tile
  [6, 100, 100, 64, true], // decoder self-attention (prompt)
  [6, 1, 64, 64, true], // decoder self-attention, one new token vs cache
  [6, 3, 50, 64, true], // a few new tokens vs cache: offset causal mask
  [6, 4, 1500, 64, false], // decoder cross-attention over encoder output
  [6, 1500, 1500, 64, false], // encoder self-attention
  [2, 33, 45, 128, false], // biggest supported head dim
];

const IMPLS = {
  unfused: attentionUnfused,
  // auto: flash for many queries, decode kernel for <= 8 (if subgroups)
  fused: attention,
  // force the query-parallel kernel even for tiny Tq
  flash: (...a: Parameters<typeof attention>) => attention(a[0], a[1], a[2], a[3], { ...a[4], kernel: "flash" }),
};

describe("decode attention kernel (key-parallel)", () => {
  // [H, Tq, Tk, D, causal]
  it.each([
    [6, 1, 1, false],
    [6, 1, 7, true],
    [6, 1, 448, true],
    [6, 3, 50, true], // offset causal mask: each query sees a different prefix
    [6, 1, 1500, false],
    [6, 8, 1500, false],
    [2, 2, 100, false, 128] as const,
    [2, 2, 100, false, 32] as const,
  ] as [number, number, number, boolean, number?][])("H=%i Tq=%i Tk=%i causal=%s D=%s", async (H, Tq, Tk, causal, D = 64) => {
    const kiln = await getKiln();
    if (!kiln.features.subgroups || kiln.features.subgroupSize !== 32) return;
    const q = randomData(H * Tq * D, 11, 1);
    const k = randomData(H * Tk * D, 12, 1);
    const v = randomData(H * Tk * D, 13, 1);
    const got = await attention(kiln, kiln.tensor(q, [H, Tq, D]), kiln.tensor(k, [H, Tk, D]), kiln.tensor(v, [H, Tk, D]), {
      causal,
      kernel: "decode",
    }).readF32();
    expectClose(got, attentionRef(q, k, v, H, Tq, Tk, D, { causal }), 1e-4, 1e-5);
  });
});

describe.each(Object.entries(IMPLS))("attention (%s)", (_name, impl) => {
  it.each(CASES)("H=%i Tq=%i Tk=%i D=%i causal=%s", async (H, Tq, Tk, D, causal) => {
    const kiln = await getKiln();
    const q = randomData(H * Tq * D, 1, 1);
    const k = randomData(H * Tk * D, 2, 1);
    const v = randomData(H * Tk * D, 3, 1);
    const got = await impl(
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
    expect(() => impl(kiln, kiln.empty([2, 3, 8]), kiln.empty([2, 4, 16]), kiln.empty([2, 4, 16]))).toThrow(/line up/);
  });
});

describe.each(["flash", "decode"] as const)("partially filled KV cache (%s kernel)", (kernel) => {
  it.each([
    [1, 5, true],
    [1, 37, false],
    [4, 37, true],
  ])("Tq=%i of %i valid keys, causal=%s", async (Tq, Tk, causal) => {
    const kiln = await getKiln();
    if (kernel === "decode" && !(kiln.features.subgroups && kiln.features.subgroupSize === 32)) return;
    const [H, D, cap] = [6, 64, 448];
    const q = randomData(H * Tq * D, 21, 1);
    const k = randomData(H * Tk * D, 22, 1);
    const v = randomData(H * Tk * D, 23, 1);
    // copy the valid rows into a full-capacity cache; fill the tail with junk
    // that would wreck the result if the kernel ever read it
    const pack = (src: Float32Array) => {
      const out = new Float32Array(H * cap * D).fill(1e6);
      for (let h = 0; h < H; h++) out.set(src.subarray(h * Tk * D, (h + 1) * Tk * D), h * cap * D);
      return out;
    };
    const got = await attention(kiln, kiln.tensor(q, [H, Tq, D]), kiln.tensor(pack(k), [H, cap, D]), kiln.tensor(pack(v), [H, cap, D]), {
      causal,
      kvLength: Tk,
      kernel,
    }).readF32();
    expectClose(got, attentionRef(q, k, v, H, Tq, Tk, D, { causal }), 1e-4, 1e-5);
  });
});

describe("fused attention limits", () => {
  it("rejects head dims it can't tile", async () => {
    const kiln = await getKiln();
    expect(() => attention(kiln, kiln.empty([1, 2, 6]), kiln.empty([1, 2, 6]), kiln.empty([1, 2, 6]))).toThrow(/head dim/);
    expect(() => attention(kiln, kiln.empty([1, 2, 256]), kiln.empty([1, 2, 256]), kiln.empty([1, 2, 256]))).toThrow(/head dim/);
  });
});
