/**
 * Device setup and feature detection.
 *
 * WebGPU hardware varies a lot more than people expect: f16 math, subgroup
 * operations and GPU timestamps are all optional. Kernels pick variants based
 * on what's here rather than assuming the best case.
 */

export interface Features {
  /** `shader-f16`: 16-bit floats in shaders (halves memory traffic). */
  f16: boolean;
  /** `subgroups`: warp-level shuffles/reductions. */
  subgroups: boolean;
  /** Subgroup width if known (32 on NVIDIA, 32/64 on AMD, varies elsewhere). */
  subgroupSize: number | null;
  /** `timestamp-query`: accurate GPU-side kernel timing. */
  timestamps: boolean;
}

export interface DeviceInfo {
  device: GPUDevice;
  features: Features;
  adapterName: string;
  limits: GPUSupportedLimits;
}

export class WebGPUUnavailableError extends Error {}

export async function requestDevice(): Promise<DeviceInfo> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    throw new WebGPUUnavailableError("WebGPU isn't available in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new WebGPUUnavailableError("No WebGPU adapter found.");

  const wanted: GPUFeatureName[] = ["shader-f16", "subgroups", "timestamp-query"];
  const requiredFeatures = wanted.filter((f) => adapter.features.has(f));

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      // Ask for everything the hardware offers. The defaults are deliberately
      // conservative (e.g. 128MB per storage binding), and model weights blow
      // straight past them.
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
    },
  });

  // Validation errors are reported asynchronously and otherwise just make a
  // dispatch silently do nothing - which looks exactly like a wrong kernel.
  device.addEventListener("uncapturederror", (e) =>
    console.error("[kiln] WebGPU error:", (e as GPUUncapturedErrorEvent).error.message),
  );

  const info = adapter.info;
  const hasSubgroups = adapter.features.has("subgroups");
  return {
    device,
    features: {
      f16: adapter.features.has("shader-f16"),
      subgroups: hasSubgroups,
      // only trust the size when min == max; otherwise kernels must not assume it
      subgroupSize:
        hasSubgroups && info?.subgroupMinSize && info.subgroupMinSize === info.subgroupMaxSize
          ? info.subgroupMinSize
          : null,
      timestamps: adapter.features.has("timestamp-query"),
    },
    adapterName: [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ") || "unknown GPU",
    limits: device.limits,
  };
}
