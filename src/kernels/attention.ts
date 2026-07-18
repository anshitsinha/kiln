import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { matmul } from "./matmul";
import { softmax } from "./softmax";

/**
 * Scaled dot-product attention.
 *
 *   q: [H, Tq, D]   k, v: [H, Tk, D]   ->   [H, Tq, D]
 */

export interface AttentionOptions {
  scale?: number;
  causal?: boolean;
}

function dims(q: Tensor, k: Tensor, v: Tensor) {
  if (q.shape.length !== 3 || k.shape.length !== 3 || v.shape.length !== 3) {
    throw new Error("attention: q, k, v must be [heads, tokens, headDim]");
  }
  const [H, Tq, D] = q.shape;
  const [Hk, Tk, Dk] = k.shape;
  if (Hk !== H || Dk !== D || v.shape[0] !== H || v.shape[1] !== Tk || v.shape[2] !== D) {
    throw new Error(`attention: shapes don't line up - q ${q.shape}, k ${k.shape}, v ${v.shape}`);
  }
  return { H, Tq, Tk, D };
}

/**
 * The unfused baseline: three kernels, the way a general graph runtime runs
 * it. QK^T writes the full [H, Tq, Tk] score matrix to memory, softmax reads
 * and rewrites it, and PV reads it again - for Whisper's encoder (6 heads,
 * 1500 tokens) that's a 54 MB intermediate, round-tripped three times.
 */
export function attentionUnfused(kiln: Kiln, q: Tensor, k: Tensor, v: Tensor, opts: AttentionOptions = {}): Tensor {
  const { D } = dims(q, k, v);
  const scores = matmul(kiln, q, k, { transposeB: true, variant: "blocked" });
  const probs = softmax(kiln, scores, { scale: opts.scale ?? 1 / Math.sqrt(D), causal: opts.causal });
  return matmul(kiln, probs, v, { variant: "blocked" });
}

// --- fused ---------------------------------------------------------------------

const BR = 64; // query rows per workgroup = threads per workgroup
const BC = 32; // keys per tile

/**
 * FlashAttention-style fused kernel: the score matrix never exists.
 *
 * Each thread owns one query row and keeps q (pre-scaled) and its output
 * accumulator in registers. The workgroup walks K/V in tiles of BC rows,
 * staged in workgroup memory - every thread reads the same key at the same
 * time, which is a broadcast, not a bank conflict. Per tile, a thread
 * computes its BC scores, then folds them into a running softmax:
 *
 *   m'   = max(m, max_j s_j)
 *   c    = e^(m - m')                  rescale what's accumulated so far
 *   l'   = l*c + sum_j e^(s_j - m')
 *   acc' = acc*c + sum_j e^(s_j - m') V_j
 *
 * and at the end writes acc / l. The rescale happens once per tile, not
 * once per key. With a causal mask, tiles entirely past the block's last
 * visible key are skipped.
 *
 * Memory traffic: Q read once, K and V read once per query block, O written
 * once - vs the unfused version's full [H, Tq, Tk] round trips.
 */
function flashWgsl(D: number, causal: boolean): string {
  const D4 = D / 4;
  return /* wgsl */ `
struct Params { Tq: u32, Tk: u32, scale: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> V: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> O: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> p: Params;

const D4 = ${D4}u;
const BR = ${BR}u;
const BC = ${BC}u;
const NEG = -3.4e38;

var<workgroup> Ks: array<vec4<f32>, ${BC * D4}>;
var<workgroup> Vs: array<vec4<f32>, ${BC * D4}>;

@compute @workgroup_size(${BR})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.y;
  let i = wid.x * BR + lid.x;          // this thread's query row
  let valid = i < p.Tq;
  let qOff = (h * p.Tq + min(i, p.Tq - 1u)) * D4;
  let kvOff = h * p.Tk * D4;

  var q: array<vec4<f32>, ${D4}>;
  var acc: array<vec4<f32>, ${D4}>;
  for (var d = 0u; d < D4; d++) {
    q[d] = Q[qOff + d] * p.scale;
    acc[d] = vec4<f32>(0.0);
  }
  var m = NEG;
  var l = 0.0;
  var s: array<f32, ${BC}>;

  ${
    causal
      ? `// last key any row in this block may see; tiles beyond it are skipped
  let blockLast = min(wid.x * BR + BR - 1u, p.Tq - 1u);
  let tkEnd = min(p.Tk, p.Tk - p.Tq + blockLast + 1u);
  let myLimit = p.Tk - p.Tq + i;       // last key this row may see`
      : `let tkEnd = p.Tk;
  let myLimit = p.Tk - 1u;`
  }

  for (var t0 = 0u; t0 < tkEnd; t0 += BC) {
    // cooperative tile load: BC * D4 vec4s each for K and V
    for (var e = lid.x; e < BC * D4; e += BR) {
      let key = t0 + e / D4;
      let inRange = key < p.Tk;
      Ks[e] = select(vec4<f32>(0.0), K[kvOff + min(key, p.Tk - 1u) * D4 + e % D4], inRange);
      Vs[e] = select(vec4<f32>(0.0), V[kvOff + min(key, p.Tk - 1u) * D4 + e % D4], inRange);
    }
    workgroupBarrier();

    if (valid) {
      var tileMax = NEG;
      for (var j = 0u; j < BC; j++) {
        let key = t0 + j;
        var sc = NEG;
        if (key < p.Tk && key <= myLimit) {
          var dotv = vec4<f32>(0.0);
          for (var d = 0u; d < D4; d++) { dotv = fma(q[d], Ks[j * D4 + d], dotv); }
          sc = dotv.x + dotv.y + dotv.z + dotv.w;
        }
        s[j] = sc;
        tileMax = max(tileMax, sc);
      }
      if (tileMax > NEG) {
        let mNew = max(m, tileMax);
        let c = exp(m - mNew);
        l *= c;
        for (var d = 0u; d < D4; d++) { acc[d] *= c; }
        for (var j = 0u; j < BC; j++) {
          // masked keys contribute exactly 0 (not e^0 when everything is NEG)
          let pj = select(0.0, exp(s[j] - mNew), s[j] > NEG);
          l += pj;
          for (var d = 0u; d < D4; d++) { acc[d] = fma(vec4<f32>(pj), Vs[j * D4 + d], acc[d]); }
        }
        m = mNew;
      }
    }
    workgroupBarrier();
  }

  if (valid) {
    let inv = 1.0 / l;
    let oOff = (h * p.Tq + i) * D4;
    for (var d = 0u; d < D4; d++) { O[oOff + d] = acc[d] * inv; }
  }
}`;
}

const flashSources = new Map<string, string>();

// --- decode: parallel over keys ("flash-decoding") -------------------------------

const SUBGROUPS = 8; // per workgroup -> 256 threads
const SG = 32;

/**
 * For a handful of queries (decoding), parallelising over query rows leaves
 * the GPU almost idle. This kernel parallelises over *keys* instead.
 *
 * One workgroup per (head, query row), 8 subgroups of 32 lanes. Keys are
 * dealt round-robin to the subgroups. Within a subgroup each lane owns D/32
 * of the head dims, so the 32 lanes load a K (or V) row together in one
 * coalesced read, and the dot product is one subgroupAdd. Each subgroup runs
 * its own online softmax - its (m, l) are subgroup-uniform, so the branch
 * on "new max?" never diverges - and at the end the 8 partial
 * (m, l, acc) triples are merged through workgroup memory:
 *
 *   M = max_g m_g,  L = sum_g l_g e^(m_g - M),  O = sum_g acc_g e^(m_g - M) / L
 *
 * Needs 32-wide subgroups and D divisible by 32.
 */
function decodeWgsl(D: number, causal: boolean): string {
  const per = D / SG; // dims per lane: 2 for D=64, 4 for D=128
  const V = per === 1 ? "f32" : `vec${per}<f32>`;
  const zero = per === 1 ? "0.0" : `${V}(0.0)`;
  const sum = per === 1 ? "x" : [..."xyzw"].slice(0, per).map((c) => `x.${c}`).join(" + ");
  return /* wgsl */ `
enable subgroups;
struct Params { Tq: u32, Tk: u32, scale: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read> Q: array<${V}>;
@group(0) @binding(1) var<storage, read> K: array<${V}>;
@group(0) @binding(2) var<storage, read> Vv: array<${V}>;
@group(0) @binding(3) var<storage, read_write> O: array<${V}>;
@group(0) @binding(4) var<uniform> p: Params;

const NEG = -3.4e38;
var<workgroup> gm: array<f32, ${SUBGROUPS}>;
var<workgroup> gl: array<f32, ${SUBGROUPS}>;
var<workgroup> gacc: array<${V}, ${SUBGROUPS * SG}>;

fn hsum(x: ${V}) -> f32 { return ${sum}; }

@compute @workgroup_size(${SUBGROUPS * SG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_invocation_id) lane: u32) {
  let i = wid.x;                 // query row
  let h = wid.y;                 // head
  let g = lid.x / ${SG}u;        // which subgroup
  let rowLen = ${SG}u;           // ${V}s per K/V/Q row (= D / ${per})
  let q = Q[(h * p.Tq + i) * rowLen + lane] * p.scale;
  let kvOff = h * p.Tk * rowLen;
  let last = ${causal ? "p.Tk - p.Tq + i" : "p.Tk - 1u"};   // last visible key

  var m = NEG;
  var l = 0.0;
  var acc = ${zero};
  // The loop has to be workgroup-uniform: subgroupAdd may only be called
  // from uniform control flow, and a bound derived from the thread id
  // ("key = g; key <= last") isn't provably uniform. So every subgroup runs
  // the same number of rounds and masks out its key past the end instead.
  for (var base = 0u; base <= last; base += ${SUBGROUPS}u) {
    let key = base + g;
    let ok = key <= last;
    let row = kvOff + min(key, last) * rowLen + lane;
    let s = subgroupAdd(hsum(q * K[row]));
    let v = Vv[row];
    if (ok) {
      if (s > m) {
        let c = exp(m - s);
        l = l * c + 1.0;
        acc = acc * c + v;
        m = s;
      } else {
        let e = exp(s - m);
        l += e;
        acc += e * v;
      }
    }
  }

  if (lane == 0u) { gm[g] = m; gl[g] = l; }
  gacc[g * ${SG}u + lane] = acc;
  workgroupBarrier();

  // first subgroup merges the ${SUBGROUPS} partials and writes the row
  if (g == 0u) {
    var M = NEG;
    for (var j = 0u; j < ${SUBGROUPS}u; j++) { M = max(M, gm[j]); }
    var L = 0.0;
    var out = ${zero};
    for (var j = 0u; j < ${SUBGROUPS}u; j++) {
      // a subgroup that saw no keys has l = 0 and contributes nothing
      let w = select(0.0, exp(gm[j] - M), gl[j] > 0.0);
      L += gl[j] * w;
      out += gacc[j * ${SG}u + lane] * w;
    }
    O[(h * p.Tq + i) * rowLen + lane] = out / L;
  }
}`;
}

/** Up to this many queries, the key-parallel decode kernel is used. */
export const DECODE_MAX_TQ = 8;

function canDecode(kiln: Kiln, Tq: number, D: number): boolean {
  return Tq <= DECODE_MAX_TQ && kiln.features.subgroups && kiln.features.subgroupSize === 32 && D % SG === 0 && D <= 128;
}

/**
 * Fused attention. Picks the key-parallel decode kernel for a few queries
 * (when the GPU has 32-wide subgroups), else the query-parallel flash
 * kernel. Head dim must be a multiple of 4 and at most 128 (the flash
 * kernel's K/V tiles have to fit in workgroup memory: 2 * BC * D * 4 bytes).
 */
export function attention(
  kiln: Kiln,
  q: Tensor,
  k: Tensor,
  v: Tensor,
  opts: AttentionOptions & { kernel?: "flash" | "decode" } = {},
): Tensor {
  const { H, Tq, Tk, D } = dims(q, k, v);
  if (D % 4 !== 0 || D > 128) throw new Error(`attention: head dim ${D} must be a multiple of 4 and <= 128`);
  const causal = opts.causal ?? false;
  if (causal && Tq > Tk) throw new Error("attention: causal needs Tq <= Tk");

  const kernel = opts.kernel ?? (canDecode(kiln, Tq, D) ? "decode" : "flash");
  if (kernel === "decode") {
    if (!canDecode(kiln, Tq, D)) throw new Error("attention: decode kernel needs 32-wide subgroups, D % 32 == 0, Tq <= 8");
    const key = `decode/${D}/${causal}`;
    let code = flashSources.get(key);
    if (!code) {
      code = decodeWgsl(D, causal);
      flashSources.set(key, code);
    }
    const out = kiln.empty([H, Tq, D]);
    const params = new ArrayBuffer(16);
    new Uint32Array(params, 0, 2).set([Tq, Tk]);
    new Float32Array(params, 8, 1)[0] = opts.scale ?? 1 / Math.sqrt(D);
    kiln.dispatch({
      label: `attention.decode${causal ? ".causal" : ""}[${H}x${Tq}x${Tk}x${D}]`,
      code,
      bindings: [
        { kind: "read", buffer: q.buffer },
        { kind: "read", buffer: k.buffer },
        { kind: "read", buffer: v.buffer },
        { kind: "write", buffer: out.buffer },
        { kind: "uniform", buffer: kiln.uniform(new Uint8Array(params)) },
      ],
      workgroups: [Tq, H],
    });
    return out;
  }

  const key = `${D}/${causal}`;
  let code = flashSources.get(key);
  if (!code) {
    code = flashWgsl(D, causal);
    flashSources.set(key, code);
  }

  const out = kiln.empty([H, Tq, D]);
  const params = new ArrayBuffer(16);
  new Uint32Array(params, 0, 2).set([Tq, Tk]);
  new Float32Array(params, 8, 1)[0] = opts.scale ?? 1 / Math.sqrt(D);
  kiln.dispatch({
    label: `attention.flash${causal ? ".causal" : ""}[${H}x${Tq}x${Tk}x${D}]`,
    code,
    bindings: [
      { kind: "read", buffer: q.buffer },
      { kind: "read", buffer: k.buffer },
      { kind: "read", buffer: v.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform(new Uint8Array(params)) },
    ],
    workgroups: [Math.ceil(Tq / BR), H],
  });
  return out;
}

export { dims as attentionDims };
