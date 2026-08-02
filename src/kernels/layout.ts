import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { grid } from "./elementwise";

/**
 * Data-movement kernels: no maths, just reshuffling. Each is one pass that
 * reads and writes every element once, one thread per output element.
 */

const WG = 256;

function oneThreadPerOutput(body: string, bindings: string): string {
  return /* wgsl */ `
${bindings}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>) {
  let i = id.x + id.y * g.x * ${WG}u;
  ${body}
}`;
}

// --- im2col --------------------------------------------------------------------

const IM2COL = oneThreadPerOutput(
  /* wgsl */ `
  let total = p.tOut * p.C * p.K;
  if (i >= total) { return; }
  let t = i / (p.C * p.K);
  let r = i % (p.C * p.K);
  let c = r / p.K;
  let j = r % p.K;
  let src = i32(t * p.stride + j) - i32(p.pad);
  var v = 0.0;
  if (src >= 0 && src < i32(p.T)) {
    v = x[c * p.strideC + u32(src) * p.strideT];
  }
  y[i] = v;`,
  /* wgsl */ `
struct Params { C: u32, T: u32, K: u32, stride: u32, pad: u32, tOut: u32, strideC: u32, strideT: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;`,
);

export interface Conv1dShape {
  C: number; // input channels
  T: number; // input length
  kernel: number;
  stride: number;
  pad: number;
  /** "CT": x is [C, T] (e.g. a mel spectrogram); "TC": x is [T, C] (activations) */
  layout: "CT" | "TC";
}

export function convOutLength({ T, kernel, stride, pad }: Conv1dShape): number {
  return Math.floor((T + 2 * pad - kernel) / stride) + 1;
}

/**
 * Unfold a 1-D signal so a convolution becomes a matmul:
 * out[t, c*K + j] = x(c, t*stride + j - pad), zero outside the signal.
 * Multiplying by a conv weight [Cout, Cin, K] viewed as [Cout, Cin*K]
 * (transposed) gives the conv output as [Tout, Cout].
 */
export function im2col(kiln: Kiln, x: Tensor, s: Conv1dShape): Tensor {
  if (x.size !== s.C * s.T) throw new Error(`im2col: x has ${x.size} elements, expected ${s.C}x${s.T}`);
  const tOut = convOutLength(s);
  const out = kiln.empty([tOut, s.C * s.kernel]);
  const [strideC, strideT] = s.layout === "CT" ? [s.T, 1] : [1, s.C];
  kiln.dispatch({
    label: `im2col[${s.C}x${s.T}->${tOut}x${s.C * s.kernel}]`,
    code: IM2COL,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([s.C, s.T, s.kernel, s.stride, s.pad, tOut, strideC, strideT]) },
    ],
    workgroups: grid(kiln, out.size),
  });
  return out;
}

// --- heads -----------------------------------------------------------------------

const SPLIT = oneThreadPerOutput(
  /* wgsl */ `
  // i indexes the *input* [T, H*D]
  if (i >= p.T * p.H * p.D) { return; }
  let t = i / (p.H * p.D);
  let hd = i % (p.H * p.D);
  let h = hd / p.D;
  let d = hd % p.D;
  y[(h * p.rows + p.rowOffset + t) * p.D + d] = x[i];`,
  /* wgsl */ `
struct Params { T: u32, H: u32, D: u32, rows: u32, rowOffset: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;`,
);

export interface SplitHeadsOptions {
  /** write into this [H, rows, D] tensor instead of a new [H, T, D] one */
  out?: Tensor;
  /** first row to write in `out` - appending to a KV cache */
  rowOffset?: number;
}

/** [T, H*D] -> [H, T, D] (or into rows rowOffset.. of a bigger [H, rows, D]). */
export function splitHeads(kiln: Kiln, x: Tensor, H: number, opts: SplitHeadsOptions = {}): Tensor {
  const [T, HD] = x.shape;
  const D = HD / H;
  if (!Number.isInteger(D)) throw new Error(`splitHeads: ${HD} isn't divisible by ${H} heads`);
  const out = opts.out ?? kiln.empty([H, T, D]);
  const rows = out.shape[1];
  const rowOffset = opts.rowOffset ?? 0;
  if (out.shape[0] !== H || out.shape[2] !== D || rowOffset + T > rows) {
    throw new Error(`splitHeads: can't write ${T} rows at ${rowOffset} into ${out.shape}`);
  }
  kiln.dispatch({
    label: `splitHeads[${T}x${H}x${D}${rowOffset ? `@${rowOffset}` : ""}]`,
    code: SPLIT,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([T, H, D, rows, rowOffset, 0, 0, 0]) },
    ],
    workgroups: grid(kiln, x.size),
  });
  return out;
}

const MERGE = oneThreadPerOutput(
  /* wgsl */ `
  // i indexes the *output* [T, H*D]
  if (i >= p.T * p.H * p.D) { return; }
  let t = i / (p.H * p.D);
  let hd = i % (p.H * p.D);
  let h = hd / p.D;
  let d = hd % p.D;
  y[i] = x[(h * p.T + t) * p.D + d];`,
  /* wgsl */ `
struct Params { T: u32, H: u32, D: u32, _a: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;`,
);

/** [H, T, D] -> [T, H*D]. */
export function mergeHeads(kiln: Kiln, x: Tensor): Tensor {
  const [H, T, D] = x.shape;
  const out = kiln.empty([T, H * D]);
  kiln.dispatch({
    label: `mergeHeads[${H}x${T}x${D}]`,
    code: MERGE,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([T, H, D, 0]) },
    ],
    workgroups: grid(kiln, out.size),
  });
  return out;
}

// --- embedding -------------------------------------------------------------------

const EMBED = oneThreadPerOutput(
  /* wgsl */ `
  if (i >= p.n * p.C) { return; }
  let r = i / p.C;
  let c = i % p.C;
  y[i] = tok[ids[r] * p.C + c] + pos[(p.posOffset + r) * p.C + c];`,
  /* wgsl */ `
struct Params { n: u32, C: u32, posOffset: u32, _a: u32 };
@group(0) @binding(0) var<storage, read> tok: array<f32>;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;`,
);

/**
 * Token + position embedding in one pass:
 * out[r] = tokTable[ids[r]] + posTable[posOffset + r].
 */
export function embed(kiln: Kiln, tokTable: Tensor, posTable: Tensor, ids: Tensor, posOffset: number): Tensor {
  const C = tokTable.shape[1];
  const n = ids.size;
  if (posOffset + n > posTable.shape[0]) throw new Error(`embed: positions ${posOffset}..${posOffset + n} exceed ${posTable.shape[0]}`);
  const out = kiln.empty([n, C]);
  kiln.dispatch({
    label: `embed[${n}x${C}]`,
    code: EMBED,
    bindings: [
      { kind: "read", buffer: tokTable.buffer },
      { kind: "read", buffer: posTable.buffer },
      { kind: "read", buffer: ids.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([n, C, posOffset, 0]) },
    ],
    workgroups: grid(kiln, out.size),
  });
  return out;
}
