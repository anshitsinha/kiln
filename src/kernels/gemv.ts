import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";

/**
 * Skinny matmul for decoding: Y[M, N] = X[M, K] @ W^T (+ bias), W stored
 * [N, K] (PyTorch Linear layout), M small (1..MAX_M).
 *
 * Why a separate kernel: generating one token is a matrix-vector product.
 * There's almost no arithmetic per byte of weight, so the only thing that
 * matters is streaming W from memory as fast as possible - the blocked
 * matmul's 64x64 tiles are pure waste at M = 1.
 *
 * Layout: each workgroup handles ROWS output columns (rows of W). Each row
 * gets LANES threads that walk K together, so consecutive threads read
 * consecutive vec4s of W - fully coalesced 16-byte loads. The LANES partial
 * sums are then combined with subgroup adds when the hardware has them
 * (one instruction per 32 lanes), else with a shared-memory tree.
 *
 * All M rows of X are processed per weight load, so a batch of up to MAX_M
 * tokens (e.g. beam search) costs about the same memory traffic as one.
 */

export const MAX_M = 8;
const LANES = 64;
const ROWS = 4; // -> 256 threads per workgroup

export interface GemvOptions {
  bias?: Tensor;
  out?: Tensor;
  /** override feature detection (tests run both paths) */
  subgroups?: boolean;
}

function gemvWgsl(vec4: boolean, hasBias: boolean, subgroups: boolean): string {
  const T = vec4 ? "vec4<f32>" : "f32";
  // K in units of T
  const kUnits = vec4 ? "p.K / 4u" : "p.K";
  const reduceDot = vec4 ? "dot" : "mul";

  const reduction = subgroups
    ? /* wgsl */ `
  // subgroupAdd sums across the 32 lanes of a subgroup in one go; each row's
  // ${LANES} lanes are ${LANES / 32} subgroups, combined through shared memory.
  for (var m = 0u; m < M; m++) {
    let s = subgroupAdd(acc[m]);
    if (sgLane == 0u) { partial[m * ${(ROWS * LANES) / 32}u + lid.x / 32u] = s; }
  }
  workgroupBarrier();
  if (lane == 0u && n < p.N) {
    for (var m = 0u; m < M; m++) {
      var total = 0.0;
      for (var i = 0u; i < ${LANES / 32}u; i++) {
        total += partial[m * ${(ROWS * LANES) / 32}u + row * ${LANES / 32}u + i];
      }
      Y[m * p.N + n] = total${hasBias ? " + bias[n]" : ""};
    }
  }`
    : /* wgsl */ `
  // portable path: tree reduction over the ${LANES} lanes of each row
  for (var m = 0u; m < M; m++) {
    partial[lid.x] = acc[m];
    workgroupBarrier();
    for (var stride = ${LANES / 2}u; stride > 0u; stride /= 2u) {
      if (lane < stride) { partial[lid.x] += partial[lid.x + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u && n < p.N) { Y[m * p.N + n] = partial[lid.x]${hasBias ? " + bias[n]" : ""}; }
    workgroupBarrier();
  }`;

  const partialSize = subgroups ? (MAX_M * ROWS * LANES) / 32 : ROWS * LANES;

  return /* wgsl */ `
${subgroups ? "enable subgroups;" : ""}
struct Params { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<${T}>;
@group(0) @binding(1) var<storage, read> W: array<${T}>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
${hasBias ? "@group(0) @binding(4) var<storage, read> bias: array<f32>;" : ""}

var<workgroup> partial: array<f32, ${partialSize}>;

fn mul(a: f32, b: f32) -> f32 { return a * b; }

@compute @workgroup_size(${ROWS * LANES})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>${subgroups ? ",\n        @builtin(subgroup_invocation_id) sgLane: u32" : ""}) {
  let row = lid.x / ${LANES}u;
  let lane = lid.x % ${LANES}u;
  let n = wid.x * ${ROWS}u + row;
  let M = p.M;
  let kUnits = ${kUnits};

  var acc: array<f32, ${MAX_M}>;
  if (n < p.N) {
    let wOff = n * kUnits;
    for (var k = lane; k < kUnits; k += ${LANES}u) {
      let w = W[wOff + k];
      for (var m = 0u; m < M; m++) {
        acc[m] += ${reduceDot}(X[m * kUnits + k], w);
      }
    }
  }
  ${reduction}
}`;
}

const sources = new Map<string, string>();

export function gemv(kiln: Kiln, x: Tensor, w: Tensor, opts: GemvOptions = {}): Tensor {
  const [N, K] = w.shape.slice(-2);
  const M = x.size / K;
  if (x.shape[x.shape.length - 1] !== K) throw new Error(`gemv: x is ${x.shape}, W is ${w.shape}`);
  if (!Number.isInteger(M) || M < 1 || M > MAX_M) throw new Error(`gemv: M must be 1..${MAX_M}, got ${M}`);
  if (x.dtype !== "f32" || w.dtype !== "f32") throw new Error("gemv: f32 only for now");

  // vec4 loads need K % 4 == 0 (true for every real model dim)
  const vec4 = K % 4 === 0;
  // the subgroup path assumes 32-wide subgroups; anything else uses the tree
  const subgroups = opts.subgroups ?? (kiln.features.subgroups && kiln.features.subgroupSize === 32);
  const key = `${vec4}/${!!opts.bias}/${subgroups}`;
  let code = sources.get(key);
  if (!code) {
    code = gemvWgsl(vec4, !!opts.bias, subgroups);
    sources.set(key, code);
  }

  const out = opts.out ?? kiln.empty([...x.shape.slice(0, -1), N]);
  kiln.dispatch({
    label: `gemv${subgroups ? ".sg" : ""}[${M}x${K}x${N}]`,
    code,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "read", buffer: w.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([M, N, K, 0]) },
      ...(opts.bias ? [{ kind: "read" as const, buffer: opts.bias.buffer }] : []),
    ],
    workgroups: [Math.ceil(N / ROWS)],
  });
  return out;
}
