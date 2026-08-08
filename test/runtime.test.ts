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
    expect(Array.from(await kept.read())).toEqual([1, 2, 3, 4]);
    // the others were destroyed: using them now is a validation error
    kiln.device.pushErrorScope("validation");
    kiln.device.queue.writeBuffer(tmp.buffer, 0, new Float32Array(1));
    kiln.device.queue.writeBuffer(inner.buffer, 0, new Float32Array(1));
    expect(await kiln.device.popErrorScope()).not.toBeNull();
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
