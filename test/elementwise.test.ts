import { describe, expect, it } from "vitest";
import { activation, add, gelu, mul } from "../src/kernels/elementwise";
import { gelu as geluRef, geluTanh, mapRef } from "../src/reference/ops";
import { expectClose, getKiln, randomData } from "./helpers";

describe("activations", () => {
  // sizes: vec4 path, scalar path (not a multiple of 4), and big enough to
  // need a 2-D dispatch grid (> 65535 * 256 elements would, so just big)
  it.each([1024, 1023, 7, 1500 * 1536])("gelu (exact) over %i elements", async (n) => {
    const kiln = await getKiln();
    const x = randomData(n, n, 6); // [-6, 6) covers both tails
    const got = await gelu(kiln, kiln.tensor(x, [n])).readF32();
    // erf approximation error is ~1.5e-7 absolute
    expectClose(got, mapRef(x, geluRef), 1e-5, 5e-6);
  });

  it("gelu_tanh and relu", async () => {
    const kiln = await getKiln();
    const x = randomData(4096, 3, 5);
    const t = await activation(kiln, kiln.tensor(x, [4096]), "gelu_tanh").readF32();
    expectClose(t, mapRef(x, geluTanh), 1e-5, 1e-6);
    const r = await activation(kiln, kiln.tensor(x, [4096]), "relu").readF32();
    expectClose(r, mapRef(x, (v) => Math.max(v, 0)), 0, 0);
  });

  it("fuses a bias add", async () => {
    const kiln = await getKiln();
    const [rows, cols] = [37, 1536];
    const x = randomData(rows * cols, 1, 3);
    const bias = randomData(cols, 2, 1);
    const got = await gelu(kiln, kiln.tensor(x, [rows, cols]), { bias: kiln.tensor(bias, [cols]) }).readF32();
    const want = mapRef(x, (v) => v).map((v, i) => geluRef(v + bias[i % cols]));
    expectClose(got, want, 1e-5, 5e-6);
  });

  it("bias with an odd last dim takes the scalar path", async () => {
    const kiln = await getKiln();
    const [rows, cols] = [5, 7];
    const x = randomData(rows * cols, 4);
    const bias = randomData(cols, 5);
    const got = await activation(kiln, kiln.tensor(x, [rows, cols]), "identity", { bias: kiln.tensor(bias, [cols]) }).readF32();
    expectClose(got, Float32Array.from(x, (v, i) => v + bias[i % cols]), 0, 1e-7);
  });

  it("rejects a bias of the wrong length", async () => {
    const kiln = await getKiln();
    expect(() => gelu(kiln, kiln.empty([4, 8]), { bias: kiln.empty([4]) })).toThrow(/bias length/);
  });
});

describe("binary ops", () => {
  it("adds same-shape tensors (residual connection)", async () => {
    const kiln = await getKiln();
    const a = randomData(3000, 1);
    const b = randomData(3000, 2);
    const got = await add(kiln, kiln.tensor(a, [3000]), kiln.tensor(b, [3000])).readF32();
    expectClose(got, Float32Array.from(a, (v, i) => v + b[i]), 0, 0);
  });

  it("broadcasts along the last dim", async () => {
    const kiln = await getKiln();
    const a = randomData(6 * 4, 3);
    const b = randomData(4, 4);
    const got = await mul(kiln, kiln.tensor(a, [6, 4]), kiln.tensor(b, [4])).readF32();
    expectClose(got, Float32Array.from(a, (v, i) => v * b[i % 4]), 0, 0);
  });
});
