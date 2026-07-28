/**
 * Mixed-radix complex FFT for any length with small prime factors.
 *
 * Whisper's STFT uses n_fft = 400, which isn't a power of two, so the usual
 * radix-2 FFT doesn't apply and a direct DFT costs n^2 = 160k complex
 * multiplies per frame (x3000 frames). 400 = 4*4*5*5, and recursive
 * Cooley-Tukey over those factors costs n * (sum of factors) ~ 7k.
 *
 * Plans (factorisation + twiddles) are cached per length.
 */

interface Plan {
  n: number;
  cos: Float64Array; // cos(2*pi*k/n)
  sin: Float64Array;
}

const plans = new Map<number, Plan>();

function plan(n: number): Plan {
  let p = plans.get(n);
  if (!p) {
    const cos = new Float64Array(n);
    const sin = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      cos[k] = Math.cos((2 * Math.PI * k) / n);
      sin[k] = Math.sin((2 * Math.PI * k) / n);
    }
    p = { n, cos, sin };
    plans.set(n, p);
  }
  return p;
}

function smallestFactor(n: number): number {
  for (const f of [4, 2, 3, 5]) if (n % f === 0) return f;
  for (let f = 7; f * f <= n; f += 2) if (n % f === 0) return f;
  return n; // prime: falls back to a direct DFT of that size
}

/**
 * In-place forward FFT (X_k = sum_j x_j e^{-2 pi i jk/n}) of re/im, reading
 * elements start, start+stride, ... - the recursion works on strided views
 * of the original arrays, writing into the out arrays.
 */
function fftRec(
  re: Float64Array,
  im: Float64Array,
  start: number,
  stride: number,
  n: number,
  outRe: Float64Array,
  outIm: Float64Array,
  outStart: number,
  root: Plan,
): void {
  if (n === 1) {
    outRe[outStart] = re[start];
    outIm[outStart] = im[start];
    return;
  }
  const p = smallestFactor(n);
  const m = n / p;
  // p sub-FFTs of length m over the decimated sequences x[r], x[r+p], ...
  for (let r = 0; r < p; r++) {
    fftRec(re, im, start + r * stride, stride * p, m, outRe, outIm, outStart + r * m, root);
  }
  // combine: X[k + q*m] = sum_r W_n^{r(k + q m)} Y_r[k]
  const step = root.n / n; // index step into the root twiddle table
  const tr = new Float64Array(p);
  const ti = new Float64Array(p);
  for (let k = 0; k < m; k++) {
    for (let r = 0; r < p; r++) {
      tr[r] = outRe[outStart + r * m + k];
      ti[r] = outIm[outStart + r * m + k];
    }
    for (let q = 0; q < p; q++) {
      const idx = k + q * m;
      let sr = 0;
      let si = 0;
      for (let r = 0; r < p; r++) {
        // e^{-2 pi i r idx / n}
        const t = ((r * idx) % n) * step;
        const c = root.cos[t];
        const s = -root.sin[t];
        sr += tr[r] * c - ti[r] * s;
        si += tr[r] * s + ti[r] * c;
      }
      outRe[outStart + idx] = sr;
      outIm[outStart + idx] = si;
    }
  }
}

/** Forward FFT of a real signal. Returns (re, im) of length n. */
export function rfft(x: ArrayLike<number>): { re: Float64Array; im: Float64Array } {
  const n = x.length;
  const re = Float64Array.from(x);
  const im = new Float64Array(n);
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  fftRec(re, im, 0, 1, n, outRe, outIm, 0, plan(n));
  return { re: outRe, im: outIm };
}

/** Direct O(n^2) DFT - the reference the FFT is tested against. */
export function dft(x: ArrayLike<number>): { re: Float64Array; im: Float64Array } {
  const n = x.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      const a = (-2 * Math.PI * ((j * k) % n)) / n;
      re[k] += x[j] * Math.cos(a);
      im[k] += x[j] * Math.sin(a);
    }
  }
  return { re, im };
}
