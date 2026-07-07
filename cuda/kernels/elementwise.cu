// TensorForge — elementwise kernels: add, mul, activations (Wave 4 / T17-T19)
//
// All kernels use a grid-stride loop so they handle arbitrary tensor sizes
// with a fixed grid size. Templated on dtype (FP32/FP16/BF16) with float
// accumulator for numerical robustness.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/CudaKernelRegistry.hpp"
#include "cuda/kernels/elementwise.cuh"
#include "tensor/Dtype.hpp"

namespace tensorforge {

// ---------------------------------------------------------------------------
// Type-conversion helpers (fp16 <-> float, bf16 <-> float).
// ---------------------------------------------------------------------------
namespace {

struct DtypeTraits;

template <typename T>
struct CudaDtypeMap;

template <>
struct CudaDtypeMap<float> { static constexpr Dtype dtype = Dtype::Float32; };

template <>
struct CudaDtypeMap<__half> { static constexpr Dtype dtype = Dtype::Float16; };

template <>
struct CudaDtypeMap<__nv_bfloat16> { static constexpr Dtype dtype = Dtype::BFloat16; };

// Convert any of fp32/fp16/bf16 -> float for accumulator arithmetic.
template <typename T>
__device__ __forceinline__ float to_float(T x);

template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

constexpr int kBlockSize = 256;

} // namespace

// ---------------------------------------------------------------------------
// Add kernel: out[i] = a[i] + b[i].
// ---------------------------------------------------------------------------
template <typename T>
__global__ void elementwise_add_kernel(const T* __restrict__ a,
                                        const T* __restrict__ b,
                                        T* __restrict__ out, int64_t n) {
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
         i < n;
         i += stride) {
        float sum = to_float(a[i]) + to_float(b[i]);
        out[i] = from_float<T>(sum);
    }
}

// ---------------------------------------------------------------------------
// Mul kernel: out[i] = a[i] * b[i].
// ---------------------------------------------------------------------------
template <typename T>
__global__ void elementwise_mul_kernel(const T* __restrict__ a,
                                        const T* __restrict__ b,
                                        T* __restrict__ out, int64_t n) {
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
         i < n;
         i += stride) {
        float prod = to_float(a[i]) * to_float(b[i]);
        out[i] = from_float<T>(prod);
    }
}

// ---------------------------------------------------------------------------
// Activation functors (T19). Elementwise framework: out[i] = op(in[i]).
// ---------------------------------------------------------------------------
struct ReluOp {
    __device__ float operator()(float x) const { return fmaxf(x, 0.0f); }
};

struct LeakyReluOp {
    float alpha;
    __device__ float operator()(float x) const {
        return x > 0.0f ? x : alpha * x;
    }
};

struct SigmoidOp {
    __device__ float operator()(float x) const {
        return 1.0f / (1.0f + __expf(-x));
    }
};

struct TanhOp {
    __device__ float operator()(float x) const { return tanhf(x); }
};

// ---------------------------------------------------------------------------
// Activation kernel: out[i] = op(in[i]).
// ---------------------------------------------------------------------------
template <typename T, typename Op>
__global__ void elementwise_unary_kernel(const T* __restrict__ in,
                                          T* __restrict__ out, int64_t n, Op op) {
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
         i < n;
         i += stride) {
        float x = to_float(in[i]);
        float y = op(x);
        out[i] = from_float<T>(y);
    }
}

namespace {

template <typename T>
void launch_add_typed(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    if (grid > 65535) grid = 65535;
    elementwise_add_kernel<T><<<grid, kBlockSize, 0, stream>>>(a, b, out, n);
}

template <typename T>
void launch_mul_typed(const T* a, const T* b, T* out, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    if (grid > 65535) grid = 65535;
    elementwise_mul_kernel<T><<<grid, kBlockSize, 0, stream>>>(a, b, out, n);
}

template <typename T, typename Op>
void launch_unary_typed(const T* in, T* out, int64_t n, cudaStream_t stream, Op op) {
    if (n <= 0) return;
    int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
    if (grid > 65535) grid = 65535;
    elementwise_unary_kernel<T><<<grid, kBlockSize, 0, stream>>>(in, out, n, op);
}

} // namespace

// ---------------------------------------------------------------------------
// Public launchers
// ---------------------------------------------------------------------------

void launch_add(const void* a, const void* b, void* out, int64_t n,
                Dtype dtype, void* stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    switch (dtype) {
    case Dtype::Float32:
        launch_add_typed(static_cast<const float*>(a),
                         static_cast<const float*>(b),
                         static_cast<float*>(out), n, cuda_stream);
        break;
    case Dtype::Float16:
        launch_add_typed(static_cast<const __half*>(a),
                         static_cast<const __half*>(b),
                         static_cast<__half*>(out), n, cuda_stream);
        break;
    case Dtype::BFloat16:
        launch_add_typed(static_cast<const __nv_bfloat16*>(a),
                         static_cast<const __nv_bfloat16*>(b),
                         static_cast<__nv_bfloat16*>(out), n, cuda_stream);
        break;
    default:
        break;
    }
}

void launch_mul(const void* a, const void* b, void* out, int64_t n,
                Dtype dtype, void* stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    switch (dtype) {
    case Dtype::Float32:
        launch_mul_typed(static_cast<const float*>(a),
                         static_cast<const float*>(b),
                         static_cast<float*>(out), n, cuda_stream);
        break;
    case Dtype::Float16:
        launch_mul_typed(static_cast<const __half*>(a),
                         static_cast<const __half*>(b),
                         static_cast<__half*>(out), n, cuda_stream);
        break;
    case Dtype::BFloat16:
        launch_mul_typed(static_cast<const __nv_bfloat16*>(a),
                         static_cast<const __nv_bfloat16*>(b),
                         static_cast<__nv_bfloat16*>(out), n, cuda_stream);
        break;
    default:
        break;
    }
}

template <typename Op>
void launch_unary(const void* in, void* out, int64_t n, Dtype dtype,
                   void* stream, Op op) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    switch (dtype) {
    case Dtype::Float32:
        launch_unary_typed(static_cast<const float*>(in),
                           static_cast<float*>(out), n, cuda_stream, op);
        break;
    case Dtype::Float16:
        launch_unary_typed(static_cast<const __half*>(in),
                           static_cast<__half*>(out), n, cuda_stream, op);
        break;
    case Dtype::BFloat16:
        launch_unary_typed(static_cast<const __nv_bfloat16*>(in),
                           static_cast<__nv_bfloat16*>(out), n, cuda_stream, op);
        break;
    default:
        break;
    }
}

void launch_relu(const void* in, void* out, int64_t n, Dtype dtype, void* stream) {
    launch_unary(in, out, n, dtype, stream, ReluOp{});
}

void launch_leaky_relu(const void* in, void* out, int64_t n, Dtype dtype,
                        float alpha, void* stream) {
    launch_unary(in, out, n, dtype, stream, LeakyReluOp{alpha});
}

void launch_sigmoid(const void* in, void* out, int64_t n, Dtype dtype, void* stream) {
    launch_unary(in, out, n, dtype, stream, SigmoidOp{});
}

void launch_tanh(const void* in, void* out, int64_t n, Dtype dtype, void* stream) {
    launch_unary(in, out, n, dtype, stream, TanhOp{});
}

template void launch_unary<ReluOp>(const void*, void*, int64_t, Dtype, void*, ReluOp);
template void launch_unary<LeakyReluOp>(const void*, void*, int64_t, Dtype, void*, LeakyReluOp);
template void launch_unary<SigmoidOp>(const void*, void*, int64_t, Dtype, void*, SigmoidOp);
template void launch_unary<TanhOp>(const void*, void*, int64_t, Dtype, void*, TanhOp);

// ---------------------------------------------------------------------------
// Function-pointer adapters so tensor/Tensor.cpp (g++) can call into us.
// ---------------------------------------------------------------------------

void add_adapter(const void* a, const void* b, void* out, int64_t n,
                 Dtype dtype, void* stream) {
    launch_add(a, b, out, n, dtype, static_cast<cudaStream_t>(stream));
}

void mul_adapter(const void* a, const void* b, void* out, int64_t n,
                 Dtype dtype, void* stream) {
    launch_mul(a, b, out, n, dtype, static_cast<cudaStream_t>(stream));
}

void relu_adapter(const void* in, void* out, int64_t n, Dtype dtype, void* stream) {
    launch_relu(in, out, n, dtype, static_cast<cudaStream_t>(stream));
}

void sigmoid_adapter(const void* in, void* out, int64_t n, Dtype dtype, void* stream) {
    launch_sigmoid(in, out, n, dtype, static_cast<cudaStream_t>(stream));
}

void tanh_adapter(const void* in, void* out, int64_t n, Dtype dtype, void* stream) {
    launch_tanh(in, out, n, dtype, static_cast<cudaStream_t>(stream));
}

void leaky_relu_adapter(const void* in, void* out, int64_t n, Dtype dtype,
                         float alpha, void* stream) {
    launch_leaky_relu(in, out, n, dtype, alpha, static_cast<cudaStream_t>(stream));
}

// Periodic refresh of the current-stream pointer so the g++ side always sees
// the latest one (the stream object is owned by CudaContext::current()).
struct StreamRegistrar {
    StreamRegistrar() {
        // Register once at startup with the initial default-stream context.
        auto& ctx = DeviceContext::current();
        register_current_stream(ctx.current_stream);
    }
    void refresh() {
        auto& ctx = DeviceContext::current();
        register_current_stream(ctx.current_stream);
    }
};

struct RegisterOnLoad {
    RegisterOnLoad() {
        register_cuda_add(add_adapter);
        register_cuda_mul(mul_adapter);
        register_cuda_relu(relu_adapter);
        register_cuda_sigmoid(sigmoid_adapter);
        register_cuda_tanh(tanh_adapter);
        register_cuda_leaky_relu(leaky_relu_adapter);

        StreamRegistrar sr;
        sr.refresh();
    }
};

__attribute__((constructor))
static void register_cuda_kernels() {
    static RegisterOnLoad reg;
    (void)reg;
}

} // namespace tensorforge