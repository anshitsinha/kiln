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

/**
 * Fused attention. Head dim must be a multiple of 4 and at most 128 (the
 * K/V tiles have to fit in workgroup memory: 2 * BC * D * 4 bytes).
 */
export function attention(kiln: Kiln, q: Tensor, k: Tensor, v: Tensor, opts: AttentionOptions = {}): Tensor {
  const { H, Tq, Tk, D } = dims(q, k, v);
  if (D % 4 !== 0 || D > 128) throw new Error(`attention: head dim ${D} must be a multiple of 4 and <= 128`);
  const causal = opts.causal ?? false;
  if (causal && Tq > Tk) throw new Error("attention: causal needs Tq <= Tk");

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
