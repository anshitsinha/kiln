import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { GROUP, type Int4Weights } from "../quant/int4";
import { LANES, MAX_M, ROWS, rowReductionScratch, rowReductionWgsl } from "./gemv";

/**
 * GEMV against int4 weights: Y[M, N] = X[M, K] @ dequant(W)^T (+ bias).
 *
 * Same shape as the f32 GEMV - LANES threads per output row, walking K
 * together for coalesced loads, then the shared row reduction - but each
 * thread now reads one u32 (8 weights) at a time instead of one vec4 (4).
 *
 * Unpacking a word is two vectorised shift+mask ops:
 *
 *   lo = (vec4(w) >> (0, 4, 8, 12))  & 0xF   -> weights k..k+3
 *   hi = (vec4(w) >> (16,20,24,28))  & 0xF   -> weights k+4..k+7
 *
 * and the math is arranged so the group scale is applied once per word,
 * not once per weight:  acc += s * (dot(x0, lo - 8) + dot(x1, hi - 8)).
 * A word never straddles groups (8 divides 32), so s is constant across it.
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

export interface GemvInt4Options {
  bias?: Tensor;
  out?: Tensor;
  subgroups?: boolean;
}

function gemvInt4Wgsl(hasBias: boolean, subgroups: boolean): string {
  return /* wgsl */ `
${subgroups ? "enable subgroups;" : ""}
struct Params { M: u32, N: u32, K: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
${hasBias ? "@group(0) @binding(5) var<storage, read> bias: array<f32>;" : ""}

var<workgroup> partial: array<f32, ${rowReductionScratch(subgroups)}>;

const SHIFT_LO = vec4<u32>(0u, 4u, 8u, 12u);
const SHIFT_HI = vec4<u32>(16u, 20u, 24u, 28u);

@compute @workgroup_size(${ROWS * LANES})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>${subgroups ? ",\n        @builtin(subgroup_invocation_id) sgLane: u32" : ""}) {
  let row = lid.x / ${LANES}u;
  let lane = lid.x % ${LANES}u;
  let n = wid.x * ${ROWS}u + row;
  let M = p.M;
  let words = p.K / 8u;           // u32s per weight row
  let groups = p.K / ${GROUP}u;   // scales per weight row
  let xVec4s = p.K / 4u;          // vec4s per X row

  var acc: array<f32, ${MAX_M}>;
  if (n < p.N) {
    for (var j = lane; j < words; j += ${LANES}u) {
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
  ${rowReductionWgsl(subgroups, hasBias)}
}`;
}

const sources = new Map<string, string>();

export function gemvInt4(kiln: Kiln, x: Tensor, w: Int4Tensor, opts: GemvInt4Options = {}): Tensor {
  const { N, K } = w;
  const M = x.size / K;
  if (x.shape[x.shape.length - 1] !== K) throw new Error(`gemvInt4: x is ${x.shape}, weights are ${N}x${K}`);
  if (!Number.isInteger(M) || M < 1 || M > MAX_M) throw new Error(`gemvInt4: M must be 1..${MAX_M}, got ${M}`);
  if (opts.bias && opts.bias.size !== N) throw new Error(`gemvInt4: bias has ${opts.bias.size} elements, N is ${N}`);

  const subgroups = opts.subgroups ?? (kiln.features.subgroups && kiln.features.subgroupSize === 32);
  const key = `${!!opts.bias}/${subgroups}`;
  let code = sources.get(key);
  if (!code) {
    code = gemvInt4Wgsl(!!opts.bias, subgroups);
    sources.set(key, code);
  }

  const out = opts.out ?? kiln.empty([...x.shape.slice(0, -1), N]);
  kiln.dispatch({
    label: `gemv.q4${subgroups ? ".sg" : ""}[${M}x${K}x${N}]`,
    code,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "read", buffer: w.packed.buffer },
      { kind: "read", buffer: w.scales.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([M, N, K, 0]) },
      ...(opts.bias ? [{ kind: "read" as const, buffer: opts.bias.buffer }] : []),
    ],
    workgroups: [Math.ceil(N / ROWS)],
  });
  return out;
}
