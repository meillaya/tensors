// TensorForge — bias_add CUDA kernel (Wave 6 / T27)
//
// Adds a per-channel bias vector to a [N, C, ...] tensor in place. Used
// by nn::Conv2d to fold the bias term into the GEMM output without
// materialising a [N, C, M] broadcast copy.
//
// Launch grid: one thread per (n, c, m) element. The bias is indexed
// from the channel axis (the second dimension of the output).

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/bias_add.cuh"
#include "tensor/Dtype.hpp"

namespace tensorforge {

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

template <typename T>
__global__ void bias_add_kernel(T* __restrict__ x,
                                 const T* __restrict__ bias,
                                 int64_t N, int64_t C, int64_t M) {
    int64_t total = N * C * M;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t c = (idx / M) % C;
    float v = to_float(x[idx]) + to_float(bias[c]);
    x[idx] = from_float<T>(v);
}

template <typename T>
void launch_bias_add_typed(T* x, const T* bias, int64_t N, int64_t C,
                            int64_t M, cudaStream_t stream) {
    if (N <= 0 || C <= 0 || M <= 0) return;
    int64_t total = N * C * M;
    int grid = static_cast<int>((total + 255) / 256);
    bias_add_kernel<T><<<grid, 256, 0, stream>>>(x, bias, N, C, M);
}

}  // namespace

void launch_bias_add(void* x, const void* bias,
                      int64_t N, int64_t C, int64_t M,
                      Dtype dtype, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (dtype) {
        case Dtype::Float32:
            launch_bias_add_typed<float>(static_cast<float*>(x),
                                          static_cast<const float*>(bias),
                                          N, C, M, s);
            break;
        case Dtype::Float16:
            launch_bias_add_typed<__half>(static_cast<__half*>(x),
                                           static_cast<const __half*>(bias),
                                           N, C, M, s);
            break;
        case Dtype::BFloat16:
            launch_bias_add_typed<__nv_bfloat16>(
                static_cast<__nv_bfloat16*>(x),
                static_cast<const __nv_bfloat16*>(bias),
                N, C, M, s);
            break;
        default:
            throw std::invalid_argument("launch_bias_add: unsupported dtype");
    }
}

}  // namespace tensorforge
