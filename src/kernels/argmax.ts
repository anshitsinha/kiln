import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";

/**
 * Index of the largest value in each row: x [rows, cols] -> u32 [rows].
 *
 * Greedy decoding needs exactly one number per step - the next token id -
 * but reading the logits back to pick it means copying all 51,865 floats
 * (~200 KB) from the GPU every token. Doing the argmax here shrinks the
 * readback to 4 bytes.
 *
 * One workgroup per row; each thread scans a strided slice, then a shared
 * memory tree merges (value, index) pairs. Ties go to the lower index, like
 * a plain left-to-right scan. `limit` restricts the search to the first
 * `limit` columns (e.g. text tokens only), with `extra` as one additional
 * allowed column (e.g. end-of-text).
 */

const WG = 256;

const WGSL = /* wgsl */ `
struct Params { cols: u32, limit: u32, extra: u32, outOffset: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> bv: array<f32, ${WG}>;
var<workgroup> bi: array<u32, ${WG}>;

fn better(v1: f32, i1: u32, v2: f32, i2: u32) -> bool {
  return v2 > v1 || (v2 == v1 && i2 < i1);
}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * p.cols;
  let t = lid.x;
  var v = -3.4e38;
  var idx = 0xffffffffu;
  for (var c = t; c < p.limit; c += ${WG}u) {
    let x0 = x[base + c];
    if (better(v, idx, x0, c)) { v = x0; idx = c; }
  }
  if (t == 0u && p.extra < p.cols) {
    let xe = x[base + p.extra];
    if (better(v, idx, xe, p.extra)) { v = xe; idx = p.extra; }
  }
  bv[t] = v;
  bi[t] = idx;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s /= 2u) {
    if (t < s && better(bv[t], bi[t], bv[t + s], bi[t + s])) {
      bv[t] = bv[t + s];
      bi[t] = bi[t + s];
    }
    workgroupBarrier();
  }
  if (t == 0u) { out[p.outOffset + wid.x] = bi[0]; }
}`;

export interface ArgmaxOptions {
  /** only consider columns 0..limit-1 (default: all) */
  limit?: number;
  /** plus this one column (default: none) */
  extra?: number;
}

export interface ArgmaxOutput {
  /** write the ids into this u32 tensor instead of a new one... */
  out?: Tensor;
  /** ...starting at this element */
  offset?: number;
}

export function argmax(kiln: Kiln, x: Tensor, opts: ArgmaxOptions & ArgmaxOutput = {}): Tensor {
  const cols = x.shape[x.shape.length - 1];
  const rows = x.size / cols;
  const out = opts.out ?? kiln.empty([rows], "u32");
  const offset = opts.offset ?? 0;
  if (out.dtype !== "u32" || offset + rows > out.size) throw new Error(`argmax: can't write ${rows} ids at ${offset} into ${out.shape} ${out.dtype}`);
  kiln.dispatch({
    label: `argmax[${rows}x${cols}]`,
    code: WGSL,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([cols, opts.limit ?? cols, opts.extra ?? 0xffffffff, offset]) },
    ],
    workgroups: [rows],
  });
  return out;
}
