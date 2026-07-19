import { describe, expect, it } from "vitest";
import { parseSafetensors, toF32, writeSafetensors } from "../src/model/safetensors";

describe("safetensors", () => {
  it("round-trips tensors through write and parse", () => {
    const a = Float32Array.from({ length: 6 }, (_, i) => i * 1.5);
    const b = Float32Array.from([-1, 2, 3.25]);
    const file = writeSafetensors([
      { name: "layer.weight", shape: [2, 3], data: a },
      { name: "layer.bias", shape: [3], data: b },
    ]);
    const parsed = parseSafetensors(file);
    expect([...parsed.keys()].sort()).toEqual(["layer.bias", "layer.weight"]);
    expect(parsed.get("layer.weight")!.shape).toEqual([2, 3]);
    expect(Array.from(toF32(parsed.get("layer.weight")!))).toEqual(Array.from(a));
    expect(Array.from(toF32(parsed.get("layer.bias")!))).toEqual(Array.from(b));
  });

  it("converts f16 and bf16", () => {
    // hand-build a file with one F16 and one BF16 tensor holding 1.0 and -2.0
    const header = JSON.stringify({
      h: { dtype: "F16", shape: [2], data_offsets: [0, 4] },
      b: { dtype: "BF16", shape: [2], data_offsets: [4, 8] },
    });
    const hb = new TextEncoder().encode(header);
    const buf = new Uint8Array(8 + hb.length + 8);
    new DataView(buf.buffer).setBigUint64(0, BigInt(hb.length), true);
    buf.set(hb, 8);
    const data = new DataView(buf.buffer, 8 + hb.length);
    data.setUint16(0, 0x3c00, true); // f16 1.0
    data.setUint16(2, 0xc000, true); // f16 -2.0
    data.setUint16(4, 0x3f80, true); // bf16 1.0
    data.setUint16(6, 0xc000, true); // bf16 -2.0
    const parsed = parseSafetensors(buf.buffer);
    expect(Array.from(toF32(parsed.get("h")!))).toEqual([1, -2]);
    expect(Array.from(toF32(parsed.get("b")!))).toEqual([1, -2]);
  });

  it("rejects corrupt files", () => {
    expect(() => parseSafetensors(new ArrayBuffer(4))).toThrow(/too small/);
    const file = new Uint8Array(writeSafetensors([{ name: "x", shape: [4], data: new Float32Array(4) }]));
    expect(() => parseSafetensors(file.slice(0, file.length - 4).buffer)).toThrow(/out of bounds/);
  });
});
