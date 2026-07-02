import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";

/**
 * LayerNorm over the last dimension: y = (x - mean) / sqrt(var + eps) * gamma + beta.
 *
 * One workgroup per row. Mean and variance come from Welford's algorithm:
 * each thread keeps (count, mean, M2) over its strided slice, then the
 * partials are merged pairwise:
 *
 *   n = na + nb,  d = mb - ma
 *   mean = ma + d * nb / n
 *   M2   = M2a + M2b + d^2 * na * nb / n
 *
 * The textbook shortcut var = E[x^2] - E[x]^2 loses most of its precision
 * in f32 when the mean is large relative to the spread - which is exactly
 * what residual-stream activations look like deeper in a network.
 */

const WG = 256;

export interface LayerNormOptions {
  eps?: number;
  out?: Tensor;
}

const WGSL = /* wgsl */ `
struct Params { cols: u32, eps: f32, _a: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

var<workgroup> wn: array<f32, ${WG}>;
var<workgroup> wmean: array<f32, ${WG}>;
var<workgroup> wm2: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x + wid.y * g.x;
  let base = row * p.cols;
  let t = lid.x;

  var n = 0.0;
  var mean = 0.0;
  var m2 = 0.0;
  for (var c = t; c < p.cols; c += ${WG}u) {
    let v = x[base + c];
    n += 1.0;
    let d = v - mean;
    mean += d / n;
    m2 += d * (v - mean);
  }

  wn[t] = n;
  wmean[t] = mean;
  wm2[t] = m2;
  workgroupBarrier();
  for (var stride = ${WG / 2}u; stride > 0u; stride /= 2u) {
    if (t < stride) {
      let na = wn[t];
      let nb = wn[t + stride];
      let nt = na + nb;
      if (nb > 0.0) {
        let d = wmean[t + stride] - wmean[t];
        wmean[t] += d * nb / nt;
        wm2[t] += wm2[t + stride] + d * d * na * nb / nt;
        wn[t] = nt;
      }
    }
    workgroupBarrier();
  }
  let rowMean = wmean[0];
  let inv = inverseSqrt(wm2[0] / f32(p.cols) + p.eps);

  for (var c = t; c < p.cols; c += ${WG}u) {
    y[base + c] = (x[base + c] - rowMean) * inv * gamma[c] + beta[c];
  }
}`;

export function layerNorm(kiln: Kiln, x: Tensor, gamma: Tensor, beta: Tensor, opts: LayerNormOptions = {}): Tensor {
  const cols = x.shape[x.shape.length - 1];
  if (gamma.size !== cols || beta.size !== cols) {
    throw new Error(`layerNorm: gamma/beta must have ${cols} elements (got ${gamma.size}, ${beta.size})`);
  }
  const rows = x.size / cols;
  const out = opts.out ?? kiln.empty(x.shape);

  const params = new ArrayBuffer(16);
  new Uint32Array(params, 0, 1)[0] = cols;
  new Float32Array(params, 4, 1)[0] = opts.eps ?? 1e-5;

  const max = kiln.device.limits.maxComputeWorkgroupsPerDimension;
  const gx = Math.min(rows, max);
  if (rows % gx !== 0) throw new Error(`layerNorm: ${rows} rows don't tile a ${gx}-wide grid`);
  kiln.dispatch({
    label: `layernorm[${rows}x${cols}]`,
    code: WGSL,
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "read", buffer: gamma.buffer },
      { kind: "read", buffer: beta.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform(new Uint8Array(params)) },
    ],
    workgroups: [gx, rows / gx],
  });
  return out;
}
