import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";

/**
 * Row-wise softmax over the last dimension, with an optional scale (fold the
 * attention 1/sqrt(d) in here instead of a separate pass) and causal mask.
 *
 * One workgroup per row. Uses the "online softmax" trick: each thread keeps
 * a running (max, sum of exp) pair over its strided slice of the row and
 * rescales the sum whenever the max grows, so max and denominator come out
 * of a single pass over the data. Then one more pass writes the output -
 * 2 reads of the row instead of the textbook 3.
 *
 * Causal masking follows the reference: in a block of `rowsPerMatrix` rows
 * and `cols` columns, row r may see columns 0 .. cols - rowsPerMatrix + r.
 * With rows == cols that's the usual lower triangle; with fewer rows than
 * columns (decoding against a KV cache) the last row sees everything.
 */

const WG = 256;

export interface SoftmaxOptions {
  scale?: number;
  causal?: boolean;
  out?: Tensor;
  /** override feature detection (tests run both reductions) */
  subgroups?: boolean;
}

// Merging (max, sum) pairs across the workgroup. The tree version is 8
// rounds of shared memory + barrier. The subgroup version merges 32 lanes per
// instruction (subgroupMax / subgroupAdd), leaving only 8 partials to combine
// through shared memory. All threads run the second-stage subgroup ops (not
// just the first subgroup) because subgroup builtins must be called from
// uniform control flow.
const TREE_REDUCE = /* wgsl */ `
  sm[t] = m;
  ss[t] = s;
  workgroupBarrier();
  for (var stride = ${WG / 2}u; stride > 0u; stride /= 2u) {
    if (t < stride) {
      let m1 = sm[t];
      let m2 = sm[t + stride];
      let mx = max(m1, m2);
      ss[t] = ss[t] * exp(m1 - mx) + ss[t + stride] * exp(m2 - mx);
      sm[t] = mx;
    }
    workgroupBarrier();
  }
  let rowMax = sm[0];
  let inv = 1.0 / ss[0];`;

const SUBGROUP_REDUCE = /* wgsl */ `
  let sgMax = subgroupMax(m);
  let sgSum = subgroupAdd(s * exp(m - sgMax));
  if (sgLane == 0u) {
    sm[t / 32u] = sgMax;
    ss[t / 32u] = sgSum;
  }
  workgroupBarrier();
  let i = sgLane;
  let pm = select(-3.4e38, sm[i], i < ${WG / 32}u);
  let ps = select(0.0, ss[i], i < ${WG / 32}u);
  let rowMax = subgroupMax(pm);
  let inv = 1.0 / subgroupAdd(ps * exp(pm - rowMax));`;

function softmaxWgsl(causal: boolean, subgroups: boolean): string {
  return /* wgsl */ `
${subgroups ? "enable subgroups;" : ""}
struct Params { cols: u32, rowsPerMatrix: u32, scale: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> sm: array<f32, ${WG}>;
var<workgroup> ss: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>${subgroups ? ",\n        @builtin(subgroup_invocation_id) sgLane: u32" : ""}) {
  let row = wid.x + wid.y * g.x;
  let base = row * p.cols;
  let t = lid.x;
  ${
    causal
      ? "let limit = i32(p.cols) - i32(p.rowsPerMatrix) + i32(row % p.rowsPerMatrix); // last visible col"
      : "let limit = i32(p.cols) - 1;"
  }

  // pass 1: per-thread online (max, sum)
  var m = -3.4e38;
  var s = 0.0;
  for (var c = t; i32(c) <= limit; c += ${WG}u) {
    let v = x[base + c] * p.scale;
    if (v > m) {
      s = s * exp(m - v) + 1.0;
      m = v;
    } else {
      s += exp(v - m);
    }
  }

  // combine across the workgroup: merging (m1, s1) and (m2, s2) gives
  // (max, s1 * e^(m1 - max) + s2 * e^(m2 - max))
  ${subgroups ? SUBGROUP_REDUCE : TREE_REDUCE}

  // pass 2: write (masked positions are exactly 0)
  for (var c = t; c < p.cols; c += ${WG}u) {
    var out = 0.0;
    if (i32(c) <= limit) { out = exp(x[base + c] * p.scale - rowMax) * inv; }
    y[base + c] = out;
  }
}`;
}

const sources = new Map<string, string>();

export function softmax(kiln: Kiln, x: Tensor, opts: SoftmaxOptions = {}): Tensor {
  const cols = x.shape[x.shape.length - 1];
  const rowsPerMatrix = x.shape.length >= 2 ? x.shape[x.shape.length - 2] : 1;
  const rows = x.size / cols;
  const causal = opts.causal ?? false;
  if (causal && rowsPerMatrix > cols) throw new Error("softmax: causal mask needs rows <= cols");

  // the subgroup path hard-codes 32-wide subgroups (t / 32 = subgroup index)
  const subgroups = opts.subgroups ?? (kiln.features.subgroups && kiln.features.subgroupSize === 32);
  const key = `${causal}/${subgroups}`;
  let code = sources.get(key);
  if (!code) {
    code = softmaxWgsl(causal, subgroups);
    sources.set(key, code);
  }

  const out = opts.out ?? kiln.empty(x.shape);
  const params = new ArrayBuffer(16);
  new Uint32Array(params, 0, 2).set([cols, rowsPerMatrix]);
  new Float32Array(params, 8, 1)[0] = opts.scale ?? 1;

  const max = kiln.device.limits.maxComputeWorkgroupsPerDimension;
  const gx = Math.min(rows, max);
  if (rows % gx !== 0) throw new Error(`softmax: ${rows} rows don't tile a ${gx}-wide grid`);
  kiln.dispatch({
    label: `softmax${causal ? ".causal" : ""}${subgroups ? ".sg" : ""}[${rows}x${cols}]`,
    code,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform(new Uint8Array(params)) },
    ],
    workgroups: [gx, rows / gx],
  });
  return out;
}
