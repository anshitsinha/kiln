export { Kiln, type BindingKind, type Dispatch, type KernelTiming } from "./gpu/kiln";
export { requestDevice, WebGPUUnavailableError, type DeviceInfo, type Features } from "./gpu/device";
export { Tensor, f16BitsToF32, f16ToF32, f32ToF16, f32ToF16Bits, numel, type DType } from "./gpu/tensor";

// kernels
export { matmul, type MatmulOptions, type MatmulVariant } from "./kernels/matmul";
export { gemv, MAX_M as GEMV_MAX_M, type GemvOptions } from "./kernels/gemv";
export { activation, add, gelu, mul, type Activation, type ActivationOptions } from "./kernels/elementwise";
export { softmax, type SoftmaxOptions } from "./kernels/softmax";
export { layerNorm, type LayerNormOptions } from "./kernels/layernorm";
