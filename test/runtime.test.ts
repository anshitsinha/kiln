import { describe, expect, it } from "vitest";
import { f16BitsToF32, f32ToF16Bits, type Tensor } from "../src";
import { getKiln, randomData } from "./helpers";

describe("runtime", () => {
  it("gets a device and reports features", async () => {
    const kiln = await getKiln();
    expect(kiln.device).toBeTruthy();
    console.log(`GPU: ${kiln.adapterName}`, kiln.features);
  });

  it("round-trips f32 and u32 tensors", async () => {
    const kiln = await getKiln();
    const data = randomData(1000, 7);
    expect(await kiln.tensor(data, [10, 100]).read()).toEqual(data);
    const ints = Uint32Array.from({ length: 33 }, (_, i) => i * 7);
    expect(await kiln.tensor(ints, [33], "u32").read()).toEqual(ints);
  });

  it("round-trips f16 tensors (odd length, so the upload needs padding)", async () => {
    const kiln = await getKiln();
    const data = [0, 1, -2, 0.5, 65504, 1e-7, 3.14159];
    const back = Array.from(await kiln.tensor(data, [7], "f16").read());
    expect(back.slice(0, 5)).toEqual([0, 1, -2, 0.5, 65504]);
    expect(back[5]).toBeCloseTo(1.19e-7, 9); // f16 subnormal
    expect(back[6]).toBeCloseTo(3.140625, 6);
  });

  it("dispatches a kernel with a uniform", async () => {
    const kiln = await getKiln();
    const x = kiln.tensor([1, 2, 3, 4, 5], [5]);
    const y = kiln.empty([5]);
    kiln.dispatch({
      label: "scale",
      code: /* wgsl */ `
        @group(0) @binding(0) var<storage, read> x: array<f32>;
        @group(0) @binding(1) var<storage, read_write> y: array<f32>;
        @group(0) @binding(2) var<uniform> p: vec4<f32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          if (id.x < arrayLength(&x)) { y[id.x] = x[id.x] * p.x; }
        }`,
      bindings: [
        { kind: "read", buffer: x.buffer },
        { kind: "write", buffer: y.buffer },
        { kind: "uniform", buffer: kiln.uniform([2.5], "f32") },
      ],
      workgroups: [1],
    });
    expect(Array.from(await y.read())).toEqual([2.5, 5, 7.5, 10, 12.5]);
  });

  it("packs many kernels' parameters into one buffer correctly", async () => {
    const kiln = await getKiln();
    const outs = [1, 2, 3, 4, 5].map((s) => {
      const y = kiln.empty([4]);
      kiln.dispatch({
        label: `fill${s}`,
        code: /* wgsl */ `
          @group(0) @binding(0) var<storage, read_write> y: array<f32>;
          @group(0) @binding(1) var<uniform> p: vec4<f32>;
          @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id: vec3<u32>) { y[id.x] = p.x; }`,
        bindings: [
          { kind: "write", buffer: y.buffer },
          { kind: "uniform", buffer: kiln.uniform([s], "f32") },
        ],
        workgroups: [1],
      });
      return y;
    });
    for (const [i, y] of outs.entries()) expect(Array.from(await y.read())).toEqual([i + 1, i + 1, i + 1, i + 1]);
  });

  it("refuses a parameter slot from before a flush", async () => {
    const kiln = await getKiln();
    const stale = kiln.uniform([1], "f32");
    kiln.dispatch({
      label: "noop",
      code: "@compute @workgroup_size(1) fn main() {}",
      bindings: [],
      workgroups: [1],
    });
    await kiln.flush(); // uploads the batch that `stale` belonged to
    kiln.dispatch({
      label: "late",
      code: "@group(0) @binding(0) var<uniform> p: vec4<f32>; @compute @workgroup_size(1) fn main() { _ = p; }",
      bindings: [{ kind: "uniform", buffer: stale }],
      workgroups: [1],
    });
    await expect(kiln.flush()).rejects.toThrow(/before a flush/);
  });

  it("scope() frees intermediates and keeps what's returned", async () => {
    const kiln = await getKiln();
    let tmp!: Tensor;
    let inner!: Tensor;
    const kept = await kiln.scope(async () => {
      tmp = kiln.empty([4]);
      const nested = await kiln.scope(() => {
        inner = kiln.empty([4]);
        return { list: [kiln.tensor([1, 2, 3, 4], [4])] };
      });
      return nested.list[0];
    });
    // the returned tensor survived both scopes
    expect(kept.released).toBe(false);
    expect(Array.from(await kept.read())).toEqual([1, 2, 3, 4]);
    // the others went back to the buffer pool and can't be used any more
    expect(tmp.released).toBe(true);
    expect(inner.released).toBe(true);
  });

  it("reuses released buffers of the same size", async () => {
    const kiln = await getKiln();
    let first!: Tensor;
    await kiln.scope(() => {
      first = kiln.tensor([9, 9, 9, 9], [4]);
    });
    expect(first.released).toBe(true);
    await expect(first.read()).rejects.toThrow(/released/);
    const again = kiln.empty([4]);
    expect(again.buffer).toBe(first.buffer); // same GPU buffer, from the pool
    first.destroy(); // must not destroy the buffer `again` now owns
    expect(Array.from(await kiln.zeros([4]).read())).toEqual([0, 0, 0, 0]);
    kiln.tensor([1, 2, 3, 4], [4]); // keep the queue busy
    expect((await again.read()).length).toBe(4);
  });

  it("times kernels with GPU timestamps when available", async () => {
    const kiln = await getKiln();
    const x = kiln.empty([1 << 20]);
    const timings = await kiln.profile(() => {
      kiln.dispatch({
        label: "fill",
        code: /* wgsl */ `
          @group(0) @binding(0) var<storage, read_write> x: array<f32>;
          @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            x[id.x] = f32(id.x);
          }`,
        bindings: [{ kind: "write", buffer: x.buffer }],
        workgroups: [(1 << 20) / 256],
      });
    });
    if (kiln.features.timestamps) {
      expect(timings).toHaveLength(1);
      expect(timings[0].ms).toBeGreaterThan(0);
    }
  });

  it("profiles a function that reads back (no empty query set)", async () => {
    const kiln = await getKiln();
    const x = kiln.zeros([256]);
    kiln.device.pushErrorScope("validation");
    let got: Float32Array | undefined;
    const timings = await kiln.profile(async () => {
      kiln.dispatch({
        label: "fill",
        code: /* wgsl */ `
          @group(0) @binding(0) var<storage, read_write> x: array<f32>;
          @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            x[id.x] = 2.0;
          }`,
        bindings: [{ kind: "write", buffer: x.buffer }],
        workgroups: [1],
      });
      got = await x.readF32();
    });
    expect(await kiln.device.popErrorScope()).toBeNull();
    expect(got![255]).toBe(2);
    if (kiln.features.timestamps) expect(timings.map((t) => t.label)).toEqual(["fill"]);
  });
});

describe("f16 conversion (CPU side)", () => {
  it("rounds to nearest even and handles specials", () => {
    expect(f16BitsToF32(f32ToF16Bits(1))).toBe(1);
    expect(f16BitsToF32(f32ToF16Bits(65504))).toBe(65504);
    expect(f16BitsToF32(f32ToF16Bits(70000))).toBe(Infinity); // overflow
    expect(f16BitsToF32(f32ToF16Bits(-Infinity))).toBe(-Infinity);
    expect(Number.isNaN(f16BitsToF32(f32ToF16Bits(NaN)))).toBe(true);
    // 1 + 2^-11 is exactly halfway between two f16 values; ties go to even (1)
    expect(f16BitsToF32(f32ToF16Bits(1 + 2 ** -11))).toBe(1);
    // the smallest f16 subnormal survives
    expect(f16BitsToF32(f32ToF16Bits(2 ** -24))).toBe(2 ** -24);
  });

  it("matches every f16 value exactly on a round trip", () => {
    for (let h = 0; h < 0x10000; h++) {
      const f = f16BitsToF32(h);
      if (Number.isNaN(f)) continue;
      const back = f32ToF16Bits(f);
      // +0 / -0 both fine; otherwise must be identical
      if (back !== h) throw new Error(`f16 0x${h.toString(16)} -> ${f} -> 0x${back.toString(16)}`);
    }
  });
});
