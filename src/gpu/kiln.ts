import { requestDevice, type DeviceInfo, type Features } from "./device";
import { BYTES, f32ToF16, numel, Tensor, type DType } from "./tensor";

/** How a kernel uses each of its buffers, in binding order. */
export type BindingKind = "read" | "write" | "uniform";

export interface Dispatch {
  /** Label shown in GPU timings and error messages. */
  label: string;
  code: string;
  bindings: { kind: BindingKind; buffer: GPUBuffer }[];
  workgroups: [number, number?, number?];
}

export interface KernelTiming {
  label: string;
  ms: number;
}

/**
 * The runtime: owns the device, compiles and caches pipelines, and batches
 * dispatches.
 *
 * Batching matters. Each queue.submit() has real CPU and driver overhead -
 * a transformer forward pass is hundreds of small kernels, and submitting
 * them one at a time can cost more than running them. So kernels are
 * *recorded* with dispatch() and only sent to the GPU on flush(), as one
 * command buffer with one compute pass.
 *
 * With profiling on, each dispatch gets its own pass bracketed by GPU
 * timestamps instead. That changes the batching (slightly), so profile runs
 * report per-kernel time but their total isn't the production number.
 */
export class Kiln {
  readonly device: GPUDevice;
  readonly features: Features;
  readonly adapterName: string;

  private pipelines = new Map<string, GPUComputePipeline>();
  private layouts = new Map<string, GPUBindGroupLayout>();
  private pending: Dispatch[] = [];
  private uniformsInFlight: GPUBuffer[] = [];

  private profiling = false;
  private timings: KernelTiming[] = [];

  // tensors allocated inside each active scope(), innermost last
  private scopes: Tensor[][] = [];

  private constructor(info: DeviceInfo) {
    this.device = info.device;
    this.features = info.features;
    this.adapterName = info.adapterName;
  }

  static async create(): Promise<Kiln> {
    return new Kiln(await requestDevice());
  }

  // ---------------------------------------------------------------------------
  // tensors

  /** Allocate an uninitialised tensor (contents are zero on WebGPU). */
  empty(shape: readonly number[], dtype: DType = "f32", label?: string): Tensor {
    const size = Math.max(4, align4(numel(shape) * BYTES[dtype]));
    const buffer = this.device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const t = new Tensor(this, buffer, shape, dtype);
    this.scopes[this.scopes.length - 1]?.push(t);
    return t;
  }

  /**
   * Run `fn` and free every tensor it allocated, except the ones it returns
   * (a Tensor, or Tensors directly inside a returned array or object -
   * including arrays one level down, like { k: Tensor[] }).
   *
   * Every kernel allocates its output, so without this a forward pass leaks
   * all its intermediates. Scopes nest; a kept tensor moves to the enclosing
   * scope, so it's still freed when that one ends. Pending work is flushed
   * before anything is destroyed.
   */
  async scope<T>(fn: () => T | Promise<T>): Promise<T> {
    this.scopes.push([]);
    let allocated: Tensor[] = [];
    let result: T;
    try {
      result = await fn();
    } finally {
      allocated = this.scopes.pop()!;
    }
    const keep = new Set<Tensor>();
    const collect = (v: unknown, depth: number) => {
      if (v instanceof Tensor) keep.add(v);
      else if (depth > 0 && Array.isArray(v)) v.forEach((x) => collect(x, depth - 1));
      else if (depth > 0 && v && typeof v === "object") Object.values(v).forEach((x) => collect(x, depth - 1));
    };
    collect(result, 2);

    await this.flush();
    const outer = this.scopes[this.scopes.length - 1];
    for (const t of allocated) {
      if (keep.has(t)) outer?.push(t);
      else t.destroy();
    }
    return result;
  }

  /** Upload data. f32 data is converted when dtype is "f16". */
  tensor(data: ArrayLike<number>, shape: readonly number[], dtype: DType = "f32", label?: string): Tensor {
    if (data.length !== numel(shape)) {
      throw new Error(`data has ${data.length} elements but shape [${shape}] needs ${numel(shape)}`);
    }
    const t = this.empty(shape, dtype, label);
    let bytes: ArrayBufferView;
    if (dtype === "f16") bytes = f32ToF16(data);
    else if (dtype === "u32") bytes = data instanceof Uint32Array ? data : Uint32Array.from(data);
    else bytes = data instanceof Float32Array ? data : Float32Array.from(data);
    this.writeBuffer(t.buffer, bytes);
    return t;
  }

  /** writeBuffer needs a 4-byte multiple; pad odd-length f16 uploads. */
  writeBuffer(buffer: GPUBuffer, data: ArrayBufferView): void {
    const len = data.byteLength;
    if (len % 4 === 0) {
      this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, len);
      return;
    }
    const padded = new Uint8Array(align4(len));
    padded.set(new Uint8Array(data.buffer, data.byteOffset, len));
    this.device.queue.writeBuffer(buffer, 0, padded);
  }

  /** Small uniform buffer for kernel parameters. Freed after the next flush. */
  uniform(values: ArrayBufferView | number[], asType: "u32" | "f32" | "i32" = "u32"): GPUBuffer {
    const data = Array.isArray(values)
      ? asType === "f32"
        ? Float32Array.from(values)
        : asType === "i32"
          ? Int32Array.from(values)
          : Uint32Array.from(values)
      : values;
    // uniform buffers must be a multiple of 16 bytes
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    const buffer = this.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    this.uniformsInFlight.push(buffer);
    return buffer;
  }

  // ---------------------------------------------------------------------------
  // dispatch

  /** Record a kernel launch. Nothing runs until flush(). */
  dispatch(d: Dispatch): void {
    this.pending.push(d);
  }

  /** Submit everything recorded so far and wait for the GPU to finish. */
  async flush(): Promise<void> {
    if (this.pending.length === 0) return;
    const batch = this.pending;
    this.pending = [];

    const encoder = this.device.createCommandEncoder();
    let query: { set: GPUQuerySet; resolve: GPUBuffer; read: GPUBuffer } | null = null;

    if (this.profiling && this.features.timestamps) {
      const count = batch.length * 2;
      query = {
        set: this.device.createQuerySet({ type: "timestamp", count }),
        resolve: this.device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
        read: this.device.createBuffer({ size: count * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      };
      batch.forEach((d, i) => {
        const pass = encoder.beginComputePass({
          label: d.label,
          timestampWrites: { querySet: query!.set, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 },
        });
        this.encode(pass, d);
        pass.end();
      });
      encoder.resolveQuerySet(query.set, 0, count, query.resolve, 0);
      encoder.copyBufferToBuffer(query.resolve, 0, query.read, 0, count * 8);
    } else {
      const pass = encoder.beginComputePass();
      for (const d of batch) this.encode(pass, d);
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    for (const b of this.uniformsInFlight) b.destroy();
    this.uniformsInFlight = [];

    if (query) {
      await query.read.mapAsync(GPUMapMode.READ);
      const ts = new BigInt64Array(query.read.getMappedRange());
      batch.forEach((d, i) => this.timings.push({ label: d.label, ms: Number(ts[i * 2 + 1] - ts[i * 2]) / 1e6 }));
      query.read.unmap();
      query.set.destroy();
      query.resolve.destroy();
      query.read.destroy();
    }
  }

  private encode(pass: GPUComputePassEncoder, d: Dispatch): void {
    const kinds = d.bindings.map((b) => b.kind);
    const pipeline = this.pipeline(d.code, kinds, d.label);
    const bindGroup = this.device.createBindGroup({
      layout: this.layout(kinds),
      entries: d.bindings.map((b, i) => ({ binding: i, resource: { buffer: b.buffer } })),
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const [x, y = 1, z = 1] = d.workgroups;
    const max = this.device.limits.maxComputeWorkgroupsPerDimension;
    if (x > max || y > max || z > max) {
      throw new Error(`${d.label}: ${x}x${y}x${z} workgroups exceeds the per-dimension limit of ${max}`);
    }
    if (x > 0 && y > 0 && z > 0) pass.dispatchWorkgroups(x, y, z);
  }

  // Explicit layouts, never "auto": "auto" drops bindings the shader doesn't
  // statically use, which turns an unused parameter into a silent validation
  // failure (learned that one the hard way in another project).
  private layout(kinds: BindingKind[]): GPUBindGroupLayout {
    const key = kinds.join(",");
    let layout = this.layouts.get(key);
    if (!layout) {
      layout = this.device.createBindGroupLayout({
        entries: kinds.map((kind, i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: kind === "uniform" ? "uniform" : kind === "read" ? "read-only-storage" : "storage" },
        })),
      });
      this.layouts.set(key, layout);
    }
    return layout;
  }

  private pipeline(code: string, kinds: BindingKind[], label: string): GPUComputePipeline {
    const key = kinds.join(",") + "\n" + code;
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ label, code });
      p = this.device.createComputePipeline({
        label,
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout(kinds)] }),
        compute: { module, entryPoint: "main" },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  // ---------------------------------------------------------------------------
  // readback + profiling

  async readBuffer(buffer: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    await this.flush(); // make sure whatever writes this buffer has run
    const size = align4(byteLength);
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0, byteLength);
    staging.unmap();
    staging.destroy();
    return copy;
  }

  /**
   * Run `fn` with per-kernel GPU timing and return the timings. Falls back to
   * an empty list when the device has no timestamp support.
   */
  async profile(fn: () => void | Promise<void>): Promise<KernelTiming[]> {
    await this.flush();
    this.profiling = true;
    this.timings = [];
    try {
      await fn();
      await this.flush();
    } finally {
      this.profiling = false;
    }
    return this.timings;
  }

  /** Check a WGSL module compiles; returns the error messages, if any. */
  async compileErrors(code: string): Promise<string[]> {
    const module = this.device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    return info.messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
  }

  destroy(): void {
    this.device.destroy();
  }
}

function align4(n: number): number {
  return Math.ceil(n / 4) * 4;
}
