import { type Epilogue, epilogueWgsl } from "./epilogue";

/**
 * Register-blocked matmul for the case that matters most - activations
 * times a Linear weight, C = act(A @ W^T + bias), with A [M, K] and W [N, K]
 * both contiguous along K - generated from a tile configuration so the
 * shape can be tuned per GPU instead of hard-coded.
 *
 * Versus the "blocked" kernel:
 *  - global loads are vec4 (K % 4 == 0, true for every real model): a
 *    quarter of the load instructions, 16-byte transactions;
 *  - the per-thread output block (TM x TN) and the K slice (BK) are
 *    parameters. A bigger block means every value read from workgroup memory
 *    feeds more FMAs; a deeper slice means fewer barriers per K.
 *
 * Thread (x, y) owns rows y + WY*i and columns x + WX*j of the tile, strided
 * rather than contiguous, so a warp's reads of workgroup memory broadcast or
 * hit consecutive words, and its stores to C are coalesced. (Contiguous
 * ownership with vec4 reads of workgroup memory was tried: ~15% slower.)
 */

export interface TileConfig {
  /** threads per workgroup along N and M */
  wx: number;
  wy: number;
  /** outputs per thread along M and N */
  tm: number;
  tn: number;
  /** K slice staged in workgroup memory per iteration (multiple of 4) */
  bk: number;
}

export const tileM = (c: TileConfig) => c.wy * c.tm;
export const tileN = (c: TileConfig) => c.wx * c.tn;

/** Why a tile can't be used (null if it can), against WebGPU's default limits. */
export function checkTile(c: TileConfig, maxSharedBytes = 16384, maxThreads = 256): string | null {
  const threads = c.wx * c.wy;
  if (threads > maxThreads) return `${threads} threads > ${maxThreads}`;
  if (c.bk % 4) return "bk must be a multiple of 4";
  const shared = c.bk * (tileM(c) + tileN(c)) * 4;
  if (shared > maxSharedBytes) return `${shared} bytes of workgroup memory > ${maxSharedBytes}`;
  if (((tileM(c) * c.bk) / 4) % threads || ((tileN(c) * c.bk) / 4) % threads) return "tile loads don't divide evenly over the threads";
  return null;
}

export function tunedMatmulWgsl(c: TileConfig, hasBias: boolean, activation: Epilogue): string {
  const err = checkTile(c);
  if (err) throw new Error(`matmul tile ${JSON.stringify(c)}: ${err}`);
  const { wx, wy, tm, tn, bk } = c;
  const BM = tileM(c);
  const BN = tileN(c);
  const T = wx * wy;
  const kv = bk / 4; // vec4s per row of a K slice
  const loadsA = (BM * kv) / T;
  const loadsB = (BN * kv) / T;

  return /* wgsl */ `
struct Params {
  M: u32, N: u32, K: u32, _pad: u32,
  strideA: u32, strideB: u32, strideC: u32, _pad2: u32,
};
@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
${hasBias ? "@group(0) @binding(4) var<storage, read> bias: array<f32>;" : ""}
${epilogueWgsl(activation)}

// k-major so the inner loop reads one row of each
var<workgroup> As: array<f32, ${bk * BM}>; // As[k * BM + m]
var<workgroup> Bs: array<f32, ${bk * BN}>; // Bs[k * BN + n]

@compute @workgroup_size(${wx}, ${wy})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let row0 = wid.y * ${BM}u;
  let col0 = wid.x * ${BN}u;
  let K4 = p.K / 4u;
  let aOff = wid.z * p.strideA / 4u;
  let bOff = wid.z * p.strideB / 4u;

  var acc: array<f32, ${tm * tn}>;
  var a: array<f32, ${tm}>;
  var b: array<f32, ${tn}>;

  for (var t = 0u; t < p.K; t += ${bk}u) {
    // stage the slices: consecutive threads take consecutive vec4s of a row
    for (var i = 0u; i < ${loadsA}u; i++) {
      let idx = li + i * ${T}u;
      let m = idx / ${kv}u;
      let k4 = idx % ${kv}u;
      let gm = row0 + m;
      let gk = t + k4 * 4u;
      let v = select(vec4<f32>(0.0), A[aOff + gm * K4 + gk / 4u], gm < p.M && gk < p.K);
      let s = k4 * 4u * ${BM}u + m;
      As[s] = v.x; As[s + ${BM}u] = v.y; As[s + ${2 * BM}u] = v.z; As[s + ${3 * BM}u] = v.w;
    }
    for (var i = 0u; i < ${loadsB}u; i++) {
      let idx = li + i * ${T}u;
      let n = idx / ${kv}u;
      let k4 = idx % ${kv}u;
      let gn = col0 + n;
      let gk = t + k4 * 4u;
      let v = select(vec4<f32>(0.0), B[bOff + gn * K4 + gk / 4u], gn < p.N && gk < p.K);
      let s = k4 * 4u * ${BN}u + n;
      Bs[s] = v.x; Bs[s + ${BN}u] = v.y; Bs[s + ${2 * BN}u] = v.z; Bs[s + ${3 * BN}u] = v.w;
    }
    workgroupBarrier();

    for (var k = 0u; k < ${bk}u; k++) {
      for (var i = 0u; i < ${tm}u; i++) { a[i] = As[k * ${BM}u + lid.y + i * ${wy}u]; }
      for (var j = 0u; j < ${tn}u; j++) { b[j] = Bs[k * ${BN}u + lid.x + j * ${wx}u]; }
      for (var i = 0u; i < ${tm}u; i++) {
        for (var j = 0u; j < ${tn}u; j++) {
          acc[i * ${tn}u + j] = fma(a[i], b[j], acc[i * ${tn}u + j]);
        }
      }
    }
    workgroupBarrier();
  }

  let cOff = wid.z * p.strideC;
  for (var i = 0u; i < ${tm}u; i++) {
    let m = row0 + lid.y + i * ${wy}u;
    if (m >= p.M) { continue; }
    for (var j = 0u; j < ${tn}u; j++) {
      let n = col0 + lid.x + j * ${wx}u;
      if (n < p.N) {
        C[cOff + m * p.N + n] = act(acc[i * ${tn}u + j]${hasBias ? " + bias[n]" : ""});
      }
    }
  }
}`;
}
