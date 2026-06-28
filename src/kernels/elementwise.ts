import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";

/**
 * Elementwise ops. These are pure memory bandwidth - one read, one write, a
 * few flops - so the only things that matter are vec4 loads and not making
 * extra passes. Hence the fused variants: `bias + GELU` as one kernel reads
 * and writes the activations once instead of twice.
 */

const WG = 256;

// WGSL has no erf. Abramowitz & Stegun 7.1.26: max abs error 1.5e-7, which
// is below f32 resolution near 1 - plenty for an activation function.
const ERF_WGSL = /* wgsl */ `
fn erf_approx(x: vec4<f32>) -> vec4<f32> {
  let s = sign(x);
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return s * (1.0 - poly * exp(-a * a));
}`;

// WGSL's builtin tanh only has to be as accurate as sinh/cosh, and on the
// NVIDIA/Chromium stack it measured ~5e-6 off - enough to fail a 1e-6 test.
// 1 - 2/(e^2x + 1) is accurate in f32 and saturates correctly at both ends
// (exp overflow -> inf -> 1, underflow -> 0 -> -1).
const TANH_WGSL = /* wgsl */ `
fn tanh_precise(x: vec4<f32>) -> vec4<f32> {
  return 1.0 - 2.0 / (exp(2.0 * x) + 1.0);
}`;

const ACTIVATIONS = {
  // exact GELU via erf - nn.GELU() default, what Whisper uses
  gelu: `0.5 * v * (1.0 + erf_approx(v * 0.7071067811865476))`,
  // tanh approximation (GPT-2 style)
  gelu_tanh: `0.5 * v * (1.0 + tanh_precise(0.7978845608028654 * (v + 0.044715 * v * v * v)))`,
  relu: `max(v, vec4<f32>(0.0))`,
  identity: `v`,
} as const;

export type Activation = keyof typeof ACTIVATIONS;

/**
 * out = act(x + bias), bias broadcast over the last dim (or absent).
 * Works on vec4s, so the total size and the bias length must be multiples of
 * 4 - true for every hidden size in practice; the scalar path handles the rest.
 */
function unaryWgsl(act: Activation, hasBias: boolean, vec4: boolean): string {
  if (!vec4) {
    // scalar fallback: reuse the vec4 math by splatting
    return /* wgsl */ `
${ERF_WGSL}
${TANH_WGSL}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: vec4<u32>; // (n, biasLen, _, _)
${hasBias ? "@group(0) @binding(3) var<storage, read> bias: array<f32>;" : ""}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>) {
  let i = id.x + id.y * g.x * ${WG}u;
  if (i >= p.x) { return; }
  let v = vec4<f32>(x[i]${hasBias ? " + bias[i % p.y]" : ""});
  y[i] = (${ACTIVATIONS[act]}).x;
}`;
  }
  return /* wgsl */ `
${ERF_WGSL}
${TANH_WGSL}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: vec4<u32>; // (n/4, biasLen/4, _, _)
${hasBias ? "@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;" : ""}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>) {
  let i = id.x + id.y * g.x * ${WG}u;
  if (i >= p.x) { return; }
  let v = x[i]${hasBias ? " + bias[i % p.y]" : ""};
  y[i] = ${ACTIVATIONS[act]};
}`;
}

function binaryWgsl(op: "+" | "*", broadcast: boolean): string {
  return /* wgsl */ `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>; // (n, bLen, _, _)
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>) {
  let i = id.x + id.y * g.x * ${WG}u;
  if (i >= p.x) { return; }
  y[i] = a[i] ${op} b[${broadcast ? "i % p.y" : "i"}];
}`;
}

const sources = new Map<string, string>();
function cached(key: string, make: () => string): string {
  let s = sources.get(key);
  if (!s) {
    s = make();
    sources.set(key, s);
  }
  return s;
}

/** 1-D element count -> 2-D workgroup grid (65535 per dimension max). */
export function grid(kiln: Kiln, threads: number, wg = WG): [number, number] {
  const groups = Math.ceil(threads / wg);
  const max = kiln.device.limits.maxComputeWorkgroupsPerDimension;
  const x = Math.min(groups, max);
  return [x, Math.ceil(groups / x)];
}

export interface ActivationOptions {
  bias?: Tensor;
  out?: Tensor;
}

/** y = act(x + bias). Pass act "identity" for a plain bias add. */
export function activation(kiln: Kiln, x: Tensor, act: Activation, opts: ActivationOptions = {}): Tensor {
  const biasLen = opts.bias?.size ?? 0;
  if (opts.bias && x.shape[x.shape.length - 1] !== biasLen) {
    throw new Error(`activation: bias length ${biasLen} doesn't match last dim of ${x.shape}`);
  }
  const vec4 = x.size % 4 === 0 && (biasLen === 0 || biasLen % 4 === 0);
  const out = opts.out ?? kiln.empty(x.shape);
  const n = vec4 ? x.size / 4 : x.size;
  kiln.dispatch({
    label: `${act}${opts.bias ? "+bias" : ""}[${x.size}]`,
    code: cached(`u/${act}/${!!opts.bias}/${vec4}`, () => unaryWgsl(act, !!opts.bias, vec4)),
    bindings: [
      { kind: "read", buffer: x.buffer },
      { kind: "write", buffer: out.buffer },
      { kind: "uniform", buffer: kiln.uniform([n, vec4 ? biasLen / 4 : biasLen, 0, 0]) },
      ...(opts.bias ? [{ kind: "read" as const, buffer: opts.bias.buffer }] : []),
    ],
    workgroups: grid(kiln, n),
  });
  return out;
}

export const gelu = (kiln: Kiln, x: Tensor, opts?: ActivationOptions) => activation(kiln, x, "gelu", opts);

function binary(kiln: Kiln, a: Tensor, b: Tensor, op: "+" | "*", out?: Tensor): Tensor {
  const broadcast = b.size !== a.size;
  if (broadcast && (a.size % b.size !== 0 || a.shape[a.shape.length - 1] % b.size !== 0)) {
    throw new Error(`${op === "+" ? "add" : "mul"}: can't broadcast ${b.shape} over ${a.shape}`);
  }
  const y = out ?? kiln.empty(a.shape);
  kiln.dispatch({
    label: `${op === "+" ? "add" : "mul"}[${a.size}]`,
    code: cached(`b/${op}/${broadcast}`, () => binaryWgsl(op, broadcast)),
    bindings: [
      { kind: "read", buffer: a.buffer },
      { kind: "read", buffer: b.buffer },
      { kind: "write", buffer: y.buffer },
      { kind: "uniform", buffer: kiln.uniform([a.size, b.size, 0, 0]) },
    ],
    workgroups: grid(kiln, a.size),
  });
  return y;
}

/** a + b, with b either the same size or broadcast along a's last dim. */
export const add = (kiln: Kiln, a: Tensor, b: Tensor, out?: Tensor) => binary(kiln, a, b, "+", out);
/** a * b, same broadcasting as add. */
export const mul = (kiln: Kiln, a: Tensor, b: Tensor, out?: Tensor) => binary(kiln, a, b, "*", out);
