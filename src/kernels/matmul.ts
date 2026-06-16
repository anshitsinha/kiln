import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";

/**
 * Batched matrix multiply: C = A @ B (+ bias).
 *
 *   A: [..., M, K]
 *   B: [K, N], or [N, K] with transposeB (PyTorch's Linear weight layout),
 *      optionally with the same leading batch dims as A
 *   C: [..., M, N]
 *
 * Three implementations, kept side by side because the benchmark comparing
 * them is the point:
 *
 *  - naive:   one thread per output element, everything from global memory.
 *  - tiled:   16x16 workgroups stage 16x16 tiles of A and B in workgroup
 *             memory, so each global load is reused 16 times.
 *  - blocked: 16x16 threads compute a 64x64 tile, 4x4 outputs per thread
 *             held in registers. Every value read from workgroup memory is
 *             reused 4 times from registers, which is what actually feeds
 *             the ALUs. This is the default.
 */

export type MatmulVariant = "naive" | "tiled" | "blocked";

export interface MatmulOptions {
  transposeB?: boolean;
  bias?: Tensor;
  variant?: MatmulVariant;
  /** write into this tensor instead of allocating one */
  out?: Tensor;
}

interface Shapes {
  M: number;
  N: number;
  K: number;
  batch: number;
  batchShape: number[];
  strideA: number;
  strideB: number; // 0 when B is shared across the batch
  strideC: number;
}

function shapesOf(a: Tensor, b: Tensor, transposeB: boolean): Shapes {
  if (a.shape.length < 2 || b.shape.length < 2) throw new Error("matmul needs at least 2-D tensors");
  const [M, K] = a.shape.slice(-2);
  const [b0, b1] = b.shape.slice(-2);
  const [kB, N] = transposeB ? [b1, b0] : [b0, b1];
  if (kB !== K) {
    throw new Error(`matmul: inner dims differ - A is ${a.shape}, B is ${b.shape}${transposeB ? " (transposed)" : ""}`);
  }
  const batchShape = a.shape.slice(0, -2) as number[];
  const batch = batchShape.reduce((x, y) => x * y, 1);
  const bBatch = b.shape.slice(0, -2).reduce((x, y) => x * y, 1);
  if (bBatch !== 1 && bBatch !== batch) throw new Error(`matmul: batch dims differ - A ${a.shape}, B ${b.shape}`);
  return { M, N, K, batch, batchShape, strideA: M * K, strideB: bBatch === 1 ? 0 : K * N, strideC: M * N };
}

const PARAMS = /* wgsl */ `
struct Params {
  M: u32, N: u32, K: u32, _pad: u32,
  strideA: u32, strideB: u32, strideC: u32, _pad2: u32,
};`;

function bindingsWgsl(hasBias: boolean): string {
  return /* wgsl */ `
${PARAMS}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
${hasBias ? "@group(0) @binding(4) var<storage, read> bias: array<f32>;" : ""}
`;
}

/** B[k, n] for either layout, as a WGSL expression. */
const bAt = (transposeB: boolean, k: string, n: string) =>
  transposeB ? `B[bOff + (${n}) * p.K + (${k})]` : `B[bOff + (${k}) * p.N + (${n})]`;

function naiveWgsl(transposeB: boolean, hasBias: boolean): string {
  return /* wgsl */ `
${bindingsWgsl(hasBias)}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let m = gid.y;
  if (m >= p.M || n >= p.N) { return; }
  let aOff = gid.z * p.strideA;
  let bOff = gid.z * p.strideB;
  var acc = 0.0;
  for (var k = 0u; k < p.K; k++) {
    acc += A[aOff + m * p.K + k] * ${bAt(transposeB, "k", "n")};
  }
  C[gid.z * p.strideC + m * p.N + n] = acc${hasBias ? " + bias[n]" : ""};
}`;
}

function tiledWgsl(transposeB: boolean, hasBias: boolean): string {
  return /* wgsl */ `
${bindingsWgsl(hasBias)}
const T = 16u;
var<workgroup> As: array<f32, 256>; // [T][T], As[r * T + k]
var<workgroup> Bs: array<f32, 256>; // [T][T], Bs[k * T + c]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wid.y * T + lid.y;
  let n = wid.x * T + lid.x;
  let aOff = wid.z * p.strideA;
  let bOff = wid.z * p.strideB;
  var acc = 0.0;

  for (var t = 0u; t < p.K; t += T) {
    // each thread loads one element of each tile (zero-padded at the edges)
    let ka = t + lid.x;
    As[lid.y * T + lid.x] = select(0.0, A[aOff + m * p.K + ka], m < p.M && ka < p.K);
    let kb = t + lid.y;
    Bs[lid.y * T + lid.x] = select(0.0, ${bAt(transposeB, "kb", "n")}, n < p.N && kb < p.K);
    workgroupBarrier();
    for (var k = 0u; k < T; k++) {
      acc += As[lid.y * T + k] * Bs[k * T + lid.x];
    }
    workgroupBarrier();
  }
  if (m < p.M && n < p.N) {
    C[wid.z * p.strideC + m * p.N + n] = acc${hasBias ? " + bias[n]" : ""};
  }
}`;
}

// 64x64 output tile per workgroup, 16-deep K slices, 4x4 outputs per thread.
const BM = 64;
const BN = 64;
const BK = 16;

function blockedWgsl(transposeB: boolean, hasBias: boolean): string {
  // Loading the B tile: coalesce along whichever dimension is contiguous in
  // memory. For [K, N] that's n; for transposed [N, K] it's k.
  const loadB = transposeB
    ? /* wgsl */ `
      let n = idx / BK;
      let k = idx % BK;`
    : /* wgsl */ `
      let k = idx / BN;
      let n = idx % BN;`;

  return /* wgsl */ `
${bindingsWgsl(hasBias)}
const BM = ${BM}u;
const BN = ${BN}u;
const BK = ${BK}u;
// Stored k-major so the inner loop reads a row of As and a row of Bs.
var<workgroup> As: array<f32, ${BM * BK}>; // As[k * BM + m]
var<workgroup> Bs: array<f32, ${BK * BN}>; // Bs[k * BN + n]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row0 = wid.y * BM;
  let col0 = wid.x * BN;
  let tid = lid.y * 16u + lid.x;
  let aOff = wid.z * p.strideA;
  let bOff = wid.z * p.strideB;

  // Each thread owns rows lid.y + 16*i and cols lid.x + 16*j (i, j in 0..4).
  // Strided rather than contiguous ownership so that neighbouring threads
  // read neighbouring workgroup-memory words (no bank conflicts) and write
  // neighbouring words of C (coalesced stores).
  var acc: array<f32, 16>;
  var a: array<f32, 4>;
  var b: array<f32, 4>;

  for (var t = 0u; t < p.K; t += BK) {
    // 1024 elements per tile, 256 threads -> 4 loads each
    for (var i = 0u; i < 4u; i++) {
      let idx = tid + i * 256u;
      let m = idx / BK;
      let k = idx % BK;
      let gm = row0 + m;
      let gk = t + k;
      As[k * BM + m] = select(0.0, A[aOff + gm * p.K + gk], gm < p.M && gk < p.K);
    }
    for (var i = 0u; i < 4u; i++) {
      let idx = tid + i * 256u;
      ${loadB}
      let gn = col0 + n;
      let gk = t + k;
      Bs[k * BN + n] = select(0.0, ${bAt(transposeB, "gk", "gn")}, gn < p.N && gk < p.K);
    }
    workgroupBarrier();

    for (var k = 0u; k < BK; k++) {
      for (var i = 0u; i < 4u; i++) { a[i] = As[k * BM + lid.y + i * 16u]; }
      for (var j = 0u; j < 4u; j++) { b[j] = Bs[k * BN + lid.x + j * 16u]; }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
          acc[i * 4u + j] = fma(a[i], b[j], acc[i * 4u + j]);
        }
      }
    }
    workgroupBarrier();
  }

  let cOff = wid.z * p.strideC;
  for (var i = 0u; i < 4u; i++) {
    let m = row0 + lid.y + i * 16u;
    if (m >= p.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = col0 + lid.x + j * 16u;
      if (n < p.N) {
        C[cOff + m * p.N + n] = acc[i * 4u + j]${hasBias ? " + bias[n]" : ""};
      }
    }
  }
}`;
}

const generators: Record<MatmulVariant, (t: boolean, b: boolean) => string> = {
  naive: naiveWgsl,
  tiled: tiledWgsl,
  blocked: blockedWgsl,
};

// The generated source depends only on (variant, transposeB, hasBias), so
// generate each once - the pipeline cache keys on source text.
const sourceCache = new Map<string, string>();

export function matmulSource(variant: MatmulVariant, transposeB: boolean, hasBias: boolean): string {
  const key = `${variant}/${transposeB}/${hasBias}`;
  let src = sourceCache.get(key);
  if (!src) {
    src = generators[variant](transposeB, hasBias);
    sourceCache.set(key, src);
  }
  return src;
}

export function matmul(kiln: Kiln, a: Tensor, b: Tensor, opts: MatmulOptions = {}): Tensor {
  const transposeB = opts.transposeB ?? false;
  const variant = opts.variant ?? "blocked";
  const s = shapesOf(a, b, transposeB);
  if (a.dtype !== "f32" || b.dtype !== "f32") throw new Error("matmul: f32 only (f16 comes later)");
  if (opts.bias && opts.bias.size !== s.N) throw new Error(`matmul: bias has ${opts.bias.size} elements, N is ${s.N}`);

  const out = opts.out ?? kiln.empty([...s.batchShape, s.M, s.N]);
  const tile = variant === "blocked" ? BM : 16;
  const tileN = variant === "blocked" ? BN : 16;

  kiln.dispatch({
    label: `matmul.${variant}${transposeB ? ".T" : ""}[${s.batch}x${s.M}x${s.K}x${s.N}]`,
    code: matmulSource(variant, transposeB, !!opts.bias),
    bindings: [
      { kind: "read", buffer: a.buffer },
      { kind: "read", buffer: b.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([s.M, s.N, s.K, 0, s.strideA, s.strideB, s.strideC, 0]) },
      ...(opts.bias ? [{ kind: "read" as const, buffer: opts.bias.buffer }] : []),
    ],
    workgroups: [Math.ceil(s.N / tileN), Math.ceil(s.M / tile), s.batch],
  });
  return out;
}
