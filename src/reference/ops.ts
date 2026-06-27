/**
 * Reference implementations of the non-matmul transformer ops. Same rules as
 * matmul.ts: obviously correct, f64 internally, slow is fine.
 */

/**
 * erf to near double precision: Taylor series for small |x|, continued
 * fraction for erfc in the tails. JS has no Math.erf, and the GPU kernels use
 * a cheaper approximation - this is what they're measured against.
 */
export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN;
  const sign = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  if (a < 3) {
    // erf(x) = 2/sqrt(pi) * sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    let term = a;
    let sum = a;
    for (let n = 1; n < 100; n++) {
      term *= (-a * a) / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    return (sign * 2 * sum) / Math.sqrt(Math.PI);
  }
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
  let f = a;
  for (let k = 60; k >= 1; k--) f = a + k / 2 / f;
  const erfc = Math.exp(-a * a) / Math.sqrt(Math.PI) / f;
  return sign * (1 - erfc);
}

/** Exact GELU, x * Phi(x) - what PyTorch's nn.GELU() (and Whisper) use. */
export function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

/** The tanh approximation (GPT-2 style), for models that use it. */
export function geluTanh(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
}

export function mapRef(x: ArrayLike<number>, f: (v: number) => number): Float32Array {
  return Float32Array.from(x as ArrayLike<number>, f);
}

/** Softmax over the last dimension (length `cols`) of a [rows, cols] array. */
export function softmaxRef(x: ArrayLike<number>, rows: number, cols: number, causal = false): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    // causal: row r may only attend to columns 0..r (offset so the last row
    // sees everything when rows < cols, as in decoding with a KV cache)
    const limit = causal ? cols - rows + r : cols - 1;
    let max = -Infinity;
    for (let c = 0; c <= limit; c++) max = Math.max(max, x[r * cols + c]);
    let sum = 0;
    for (let c = 0; c <= limit; c++) sum += Math.exp(x[r * cols + c] - max);
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = c <= limit ? Math.exp(x[r * cols + c] - max) / sum : 0;
    }
  }
  return out;
}

/** LayerNorm over the last dimension, biased variance like PyTorch. */
export function layerNormRef(
  x: ArrayLike<number>,
  rows: number,
  cols: number,
  gamma: ArrayLike<number>,
  beta: ArrayLike<number>,
  eps = 1e-5,
): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += x[r * cols + c];
    mean /= cols;
    let variance = 0;
    for (let c = 0; c < cols; c++) variance += (x[r * cols + c] - mean) ** 2;
    variance /= cols;
    const inv = 1 / Math.sqrt(variance + eps);
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = (x[r * cols + c] - mean) * inv * gamma[c] + beta[c];
    }
  }
  return out;
}
