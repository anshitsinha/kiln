import { f16BitsToF32 } from "../gpu/tensor";

/**
 * Minimal safetensors reader.
 *
 * Format: an 8-byte little-endian header length N, N bytes of JSON mapping
 * tensor names to { dtype, shape, data_offsets: [begin, end] } (offsets
 * relative to the end of the header), then the raw tensor bytes. That's the
 * whole spec - no pickling, which is why it's safe to load in a browser.
 */

export type SafeDType = "F32" | "F16" | "BF16" | "I32" | "I64" | "U8";

export interface SafeTensor {
  name: string;
  dtype: SafeDType;
  shape: number[];
  /** the raw bytes, a view into the file buffer (no copy) */
  bytes: Uint8Array;
}

export class SafetensorsError extends Error {}

export function parseSafetensors(buffer: ArrayBuffer): Map<string, SafeTensor> {
  if (buffer.byteLength < 8) throw new SafetensorsError("file too small to be safetensors");
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  if (8 + headerLen > buffer.byteLength) throw new SafetensorsError(`header length ${headerLen} runs past the end of the file`);

  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen))) as Record<
    string,
    { dtype: SafeDType; shape: number[]; data_offsets: [number, number] }
  >;
  const dataStart = 8 + headerLen;
  const tensors = new Map<string, SafeTensor>();

  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [begin, end] = info.data_offsets;
    if (dataStart + end > buffer.byteLength || begin > end) {
      throw new SafetensorsError(`${name}: data offsets [${begin}, ${end}] are out of bounds`);
    }
    const expected = info.shape.reduce((a, b) => a * b, 1) * bytesPer(info.dtype);
    if (end - begin !== expected) {
      throw new SafetensorsError(`${name}: ${end - begin} bytes for shape [${info.shape}] ${info.dtype} (expected ${expected})`);
    }
    tensors.set(name, { name, dtype: info.dtype, shape: info.shape, bytes: new Uint8Array(buffer, dataStart + begin, end - begin) });
  }
  return tensors;
}

function bytesPer(dtype: SafeDType): number {
  switch (dtype) {
    case "F32":
    case "I32":
      return 4;
    case "F16":
    case "BF16":
      return 2;
    case "I64":
      return 8;
    case "U8":
      return 1;
    default:
      throw new SafetensorsError(`unsupported dtype ${dtype as string}`);
  }
}

/** A tensor's values as f32, converting from f16/bf16 if needed. */
export function toF32(t: SafeTensor): Float32Array {
  // the data may not be 4-byte aligned inside the file, so copy rather than
  // view whenever alignment isn't guaranteed
  switch (t.dtype) {
    case "F32":
      return t.bytes.byteOffset % 4 === 0
        ? new Float32Array(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength / 4)
        : new Float32Array(t.bytes.slice().buffer);
    case "F16": {
      const h = new Uint16Array(t.bytes.slice().buffer);
      return Float32Array.from(h, f16BitsToF32);
    }
    case "BF16": {
      // bf16 is the top half of an f32
      const h = new Uint16Array(t.bytes.slice().buffer);
      const out = new Uint32Array(h.length);
      for (let i = 0; i < h.length; i++) out[i] = h[i] << 16;
      return new Float32Array(out.buffer);
    }
    default:
      throw new SafetensorsError(`${t.name}: can't convert ${t.dtype} to f32`);
  }
}

/** Build a safetensors file (used by tests, handy for fixtures). */
export function writeSafetensors(tensors: { name: string; shape: number[]; data: Float32Array }[]): ArrayBuffer {
  let offset = 0;
  const header: Record<string, unknown> = {};
  for (const t of tensors) {
    header[t.name] = { dtype: "F32", shape: t.shape, data_offsets: [offset, offset + t.data.byteLength] };
    offset += t.data.byteLength;
  }
  let json = JSON.stringify(header);
  // pad the header so tensor data starts 8-byte aligned, like real files
  while ((8 + json.length) % 8 !== 0) json += " ";
  const headerBytes = new TextEncoder().encode(json);
  const out = new Uint8Array(8 + headerBytes.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  out.set(headerBytes, 8);
  let pos = 8 + headerBytes.length;
  for (const t of tensors) {
    out.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), pos);
    pos += t.data.byteLength;
  }
  return out.buffer;
}
