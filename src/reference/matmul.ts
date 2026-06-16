/**
 * Plain-JS reference implementations. These are the spec the GPU kernels are
 * tested against: obviously correct, accumulate in f64, no cleverness.
 */

export interface MatmulRefOptions {
  /** B is stored [N, K] (PyTorch Linear weight layout) instead of [K, N]. */
  transposeB?: boolean;
  /** Added to every row of the output, length N. */
  bias?: ArrayLike<number>;
}

/** C[M, N] = A[M, K] @ B (+ bias), one matrix. */
export function matmulRef(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  M: number,
  N: number,
  K: number,
  opts: MatmulRefOptions = {},
): Float32Array {
  const c = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let acc = 0;
      for (let k = 0; k < K; k++) {
        const bv = opts.transposeB ? b[n * K + k] : b[k * N + n];
        acc += a[m * K + k] * bv;
      }
      c[m * N + n] = acc + (opts.bias ? opts.bias[n] : 0);
    }
  }
  return c;
}
