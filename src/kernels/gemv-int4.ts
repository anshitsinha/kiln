import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { GROUP, type Int4Weights } from "../quant/int4";
import { type Epilogue, epilogueWgsl } from "./epilogue";
import { MAX_M } from "./gemv";

/**
 * GEMV against int4 weights: Y[M, N] = X[M, K] @ dequant(W)^T (+ bias).
 *
 * Each output row gets `lanes` threads that walk its packed weights together
 * (coalesced u32 loads), and a 256-thread workgroup handles 256 / lanes rows.
 *
 * Unpacking a word is two vectorised shift+mask ops:
 *
 *   lo = (vec4(w) >> (0, 4, 8, 12))  & 0xF   -> weights k..k+3
 *   hi = (vec4(w) >> (16,20,24,28))  & 0xF   -> weights k+4..k+7
 *
 * and the group scale is applied once per word, not once per weight:
 * acc += s * (dot(x0, lo - 8) + dot(x1, hi - 8)). A word never straddles
 * groups (8 divides 32), so s is constant across it.
 *
 * Why `lanes` is a knob: an int4 row is short. At K = 384 it's 48 words, so
 * the 64 lanes the f32 GEMV uses leave a quarter of the threads idle and the
 * rest doing a single load before a 64-way reduction - all overhead. Fewer
 * lanes per row means more loads per thread and a cheaper reduction. With
 * lanes <= 32 a row's lanes sit inside one 32-wide subgroup, so the sum is a
 * butterfly of subgroupShuffleXor - log2(lanes) register shuffles, no
 * workgroup memory, no barriers.
 */

export interface Int4Tensor {
  packed: Tensor; // u32 [N, K/8]
  scales: Tensor; // f32 [N, K/GROUP]
  N: number;
  K: number;
}

export function uploadInt4(kiln: Kiln, q: Int4Weights, label?: string): Int4Tensor {
  return {
    packed: kiln.tensor(q.packed, [q.N, q.K / 8], "u32", label && `${label}.q4`),
    scales: kiln.tensor(q.scales, [q.N, q.K / GROUP], "f32", label && `${label}.scales`),
    N: q.N,
    K: q.K,
  };
}

export type Lanes = 4 | 8 | 16 | 32 | 64;

export interface GemvInt4Options {
  bias?: Tensor;
  out?: Tensor;
  subgroups?: boolean;
  /** applied to each output (after the bias) before it's stored */
  activation?: Epilogue;
  /** threads per output row; default picked from K */
  lanes?: Lanes;
}

const WG = 256;

/**
 * Threads per row, from a sweep on the decode benchmark (NVIDIA Turing):
 *
 *   N = 384    best 16 lanes   (K = 384 and K = 1536 alike)
 *   N = 1536   best 8
 *   N = 51865  best 4
 *
 * It tracks N, not K: what matters is keeping enough threads in flight.
 * Small layers need more lanes per row just to fill the GPU; big layers
 * have rows to spare and do better with fewer, busier lanes. Tuned on one
 * GPU, so treat the thresholds as a starting point.
 */
export function defaultLanes(N: number, K: number): Lanes {
  let lanes: Lanes = N >= 16384 ? 4 : N >= 1024 ? 8 : 16;
  // never more lanes than there are words in a row
  while (lanes > 4 && lanes > K / 8) lanes = (lanes / 2) as Lanes;
  return lanes;
}

function gemvInt4Wgsl(hasBias: boolean, subgroups: boolean, lanes: Lanes, activation: Epilogue): string {
  const rows = WG / lanes;
  const shuffle = subgroups && lanes <= 32;

  const reduction = shuffle
    ? /* wgsl */ `
  // butterfly within each row's ${lanes} lanes (all inside one subgroup)
  for (var m = 0u; m < M; m++) {
    var v = acc[m];
    ${[...Array(Math.log2(lanes)).keys()]
      .map((i) => `v += subgroupShuffleXor(v, ${lanes >> (i + 1)}u);`)
      .join("\n    ")}
    if (lane == 0u && n < p.N) { Y[m * p.N + n] = act(v${hasBias ? " + bias[n]" : ""}); }
  }`
    : /* wgsl */ `
  // portable: tree over each row's ${lanes} lanes in workgroup memory
  for (var m = 0u; m < M; m++) {
    partial[lid.x] = acc[m];
    workgroupBarrier();
    for (var stride = ${lanes / 2}u; stride > 0u; stride /= 2u) {
      if (lane < stride) { partial[lid.x] += partial[lid.x + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u && n < p.N) { Y[m * p.N + n] = act(partial[lid.x]${hasBias ? " + bias[n]" : ""}); }
    workgroupBarrier();
  }`;

  return /* wgsl */ `
${shuffle ? "enable subgroups;" : ""}
struct Params { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
${hasBias ? "@group(0) @binding(5) var<storage, read> bias: array<f32>;" : ""}
${shuffle ? "" : `var<workgroup> partial: array<f32, ${WG}>;`}

const SHIFT_LO = vec4<u32>(0u, 4u, 8u, 12u);
const SHIFT_HI = vec4<u32>(16u, 20u, 24u, 28u);
${epilogueWgsl(activation)}

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = lid.x / ${lanes}u;
  let lane = lid.x % ${lanes}u;
  let n = wid.x * ${rows}u + row;
  let M = p.M;
  let words = p.K / 8u;           // u32s per weight row
  let groups = p.K / ${GROUP}u;   // scales per weight row
  let xVec4s = p.K / 4u;          // vec4s per X row

  var acc: array<f32, ${MAX_M}>;
  if (n < p.N) {
    for (var j = lane; j < words; j += ${lanes}u) {
      let w = W[n * words + j];
      let s = S[n * groups + j / ${GROUP / 8}u];
      let lo = vec4<f32>((vec4<u32>(w) >> SHIFT_LO) & vec4<u32>(15u)) - 8.0;
      let hi = vec4<f32>((vec4<u32>(w) >> SHIFT_HI) & vec4<u32>(15u)) - 8.0;
      for (var m = 0u; m < M; m++) {
        let xi = m * xVec4s + 2u * j;
        acc[m] += s * (dot(X[xi], lo) + dot(X[xi + 1u], hi));
      }
    }
  }
  ${reduction}
}`;
}

const sources = new Map<string, string>();

export function gemvInt4(kiln: Kiln, x: Tensor, w: Int4Tensor, opts: GemvInt4Options = {}): Tensor {
  const { N, K } = w;
  const M = x.size / K;
  if (x.shape[x.shape.length - 1] !== K) throw new Error(`gemvInt4: x is ${x.shape}, weights are ${N}x${K}`);
  if (!Number.isInteger(M) || M < 1 || M > MAX_M) throw new Error(`gemvInt4: M must be 1..${MAX_M}, got ${M}`);
  if (opts.bias && opts.bias.size !== N) throw new Error(`gemvInt4: bias has ${opts.bias.size} elements, N is ${N}`);

  const lanes = opts.lanes ?? defaultLanes(N, K);
  // shuffles assume a row's lanes share one 32-wide subgroup
  const subgroups = opts.subgroups ?? (kiln.features.subgroups && kiln.features.subgroupSize === 32);
  const activation = opts.activation ?? "none";
  const key = `${!!opts.bias}/${subgroups}/${lanes}/${activation}`;
  let code = sources.get(key);
  if (!code) {
    code = gemvInt4Wgsl(!!opts.bias, subgroups, lanes, activation);
    sources.set(key, code);
  }

  const out = opts.out ?? kiln.empty([...x.shape.slice(0, -1), N]);
  const rows = WG / lanes;
  kiln.dispatch({
    label: `gemv.q4.l${lanes}${subgroups && lanes <= 32 ? ".sg" : ""}${activation === "none" ? "" : `+${activation}`}[${M}x${K}x${N}]`,
    code,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "read", buffer: w.packed.buffer },
      { kind: "read", buffer: w.scales.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([M, N, K, 0]) },
      ...(opts.bias ? [{ kind: "read" as const, buffer: opts.bias.buffer }] : []),
    ],
    workgroups: [Math.ceil(N / rows)],
  });
  return out;
}
