import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { grid } from "../kernels/elementwise";
import { matmul } from "../kernels/matmul";
import { CHUNK_SAMPLES, HOP, melFilters, N_FFT, N_FRAMES, N_MELS } from "./mel";

/**
 * The log-mel spectrogram on the GPU, same maths as logMelSpectrogram():
 *
 *   frames  [3000, 400]  windowed slices of the (reflect-padded) audio
 *   spectra [3000, 402]  = frames @ basis^T, basis rows = cos / -sin of the
 *                          201 DFT bins - the FFT replaced by one matmul
 *   power   [3000, 201]  re^2 + im^2
 *   mel     [3000, 80]   = power @ filters^T
 *   log10, global max, floor at max - 8, (x + 4) / 4
 *
 * A direct DFT is O(n^2) per frame where the FFT is O(n log n), but as a
 * matmul it runs on the fastest kernel in the library, and at n = 400 that
 * wins easily. Output is [3000, 80] (time x mels) - the "TC" layout the
 * encoder's im2col reads directly, so no transpose.
 */

const N_BINS = N_FFT / 2 + 1;
const WG = 256;

const FRAMES = /* wgsl */ `
@group(0) @binding(0) var<storage, read> audio: array<f32>;
@group(0) @binding(1) var<storage, read_write> frames: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>) {
  let i = id.x + id.y * g.x * ${WG}u;
  if (i >= ${N_FRAMES * N_FFT}u) { return; }
  let t = i / ${N_FFT}u;
  let k = i % ${N_FFT}u;
  // periodic Hann window
  let w = 0.5 - 0.5 * cos(6.283185307179586 * f32(k) / ${N_FFT}.0);
  frames[i] = audio[t * ${HOP}u + k] * w;
}`;

const POWER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> spec: array<f32>;   // [frames, 2 * bins]: re then im
@group(0) @binding(1) var<storage, read_write> power: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>) {
  let i = id.x + id.y * g.x * ${WG}u;
  if (i >= ${N_FRAMES * N_BINS}u) { return; }
  let t = i / ${N_BINS}u;
  let k = i % ${N_BINS}u;
  let re = spec[t * ${2 * N_BINS}u + k];
  let im = spec[t * ${2 * N_BINS}u + ${N_BINS}u + k];
  power[i] = re * re + im * im;
}`;

// log10 in place, and the max of the result - one workgroup, it's 240k values
const LOG_MAX = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> mel: array<f32>;
@group(0) @binding(1) var<storage, read_write> maxOut: array<f32>;
var<workgroup> m: array<f32, ${WG}>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var best = -3.4e38;
  for (var i = lid.x; i < ${N_FRAMES * N_MELS}u; i += ${WG}u) {
    let v = log(max(mel[i], 1e-10)) * 0.4342944819032518; // log10
    mel[i] = v;
    best = max(best, v);
  }
  m[lid.x] = best;
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s /= 2u) {
    if (lid.x < s) { m[lid.x] = max(m[lid.x], m[lid.x + s]); }
    workgroupBarrier();
  }
  if (lid.x == 0u) { maxOut[0] = m[0]; }
}`;

const NORMALIZE = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> mel: array<f32>;
@group(0) @binding(1) var<storage, read> maxIn: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) g: vec3<u32>) {
  let i = id.x + id.y * g.x * ${WG}u;
  if (i >= ${N_FRAMES * N_MELS}u) { return; }
  mel[i] = (max(mel[i], maxIn[0] - 8.0) + 4.0) / 4.0;
}`;

/** The constant matrices, uploaded once per Kiln. */
interface MelConstants {
  basis: Tensor; // [2 * bins, n_fft]: cos rows then -sin rows
  filters: Tensor; // [mels, bins]
}

const constants = new WeakMap<Kiln, MelConstants>();

function melConstants(kiln: Kiln): MelConstants {
  let c = constants.get(kiln);
  if (!c) {
    const basis = new Float32Array(2 * N_BINS * N_FFT);
    for (let k = 0; k < N_BINS; k++) {
      for (let n = 0; n < N_FFT; n++) {
        // reduce k*n mod N first so the angle stays small and exact
        const a = (2 * Math.PI * ((k * n) % N_FFT)) / N_FFT;
        basis[k * N_FFT + n] = Math.cos(a);
        basis[(N_BINS + k) * N_FFT + n] = -Math.sin(a);
      }
    }
    // not inside any scope: these live as long as the Kiln
    c = {
      basis: kiln.tensor(basis, [2 * N_BINS, N_FFT], "f32", "mel.dft_basis"),
      filters: kiln.tensor(melFilters(), [N_MELS, N_BINS], "f32", "mel.filters"),
    };
    constants.set(kiln, c);
  }
  return c;
}

/** numpy "reflect" padding, as in mel.ts. */
function reflectPad(x: Float32Array, pad: number): Float32Array {
  const out = new Float32Array(x.length + 2 * pad);
  out.set(x, pad);
  for (let i = 0; i < pad; i++) {
    out[pad - 1 - i] = x[i + 1];
    out[pad + x.length + i] = x[x.length - 2 - i];
  }
  return out;
}

/**
 * Up to 30 s of 16 kHz mono audio -> log-mel [3000, 80] on the GPU.
 * Call inside kiln.scope() to free the intermediates.
 */
export function logMelSpectrogramGpu(kiln: Kiln, samples: Float32Array): Tensor {
  const { basis, filters } = melConstants(kiln);
  const audio = new Float32Array(CHUNK_SAMPLES);
  audio.set(samples.subarray(0, CHUNK_SAMPLES));
  const padded = kiln.tensor(reflectPad(audio, N_FFT / 2), [CHUNK_SAMPLES + N_FFT]);

  const frames = kiln.empty([N_FRAMES, N_FFT]);
  kiln.dispatch({
    label: "mel.frames",
    code: FRAMES,
    bindings: [
      { kind: "read", buffer: padded.buffer },
      { kind: "write", buffer: frames.buffer },
    ],
    workgroups: grid(kiln, N_FRAMES * N_FFT),
  });

  const spec = matmul(kiln, frames, basis, { transposeB: true, variant: "blocked" }); // [3000, 402]
  const power = kiln.empty([N_FRAMES, N_BINS]);
  kiln.dispatch({
    label: "mel.power",
    code: POWER,
    bindings: [
      { kind: "read", buffer: spec.buffer },
      { kind: "write", buffer: power.buffer },
    ],
    workgroups: grid(kiln, N_FRAMES * N_BINS),
  });

  const mel = matmul(kiln, power, filters, { transposeB: true, variant: "blocked" }); // [3000, 80]
  const max = kiln.empty([1]);
  kiln.dispatch({
    label: "mel.log_max",
    code: LOG_MAX,
    bindings: [
      { kind: "write", buffer: mel.buffer },
      { kind: "write", buffer: max.buffer },
    ],
    workgroups: [1],
  });
  kiln.dispatch({
    label: "mel.normalize",
    code: NORMALIZE,
    bindings: [
      { kind: "write", buffer: mel.buffer },
      { kind: "read", buffer: max.buffer },
    ],
    workgroups: grid(kiln, N_FRAMES * N_MELS),
  });
  return mel;
}
