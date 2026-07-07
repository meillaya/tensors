// TensorForge — layer-norm kernel implementation (Wave 4 / T21)
//
// Two-pass layer normalization: each row of a [rows, cols] input tensor is
// normalized to zero mean / unit variance, then affine-transformed by
// per-column gamma/beta.
//
// Strategy: ONE block per row. Two passes inside the kernel.
//
//   pass 1 — sum over the row, then sum of squares. Block-reduce both.
//            mean = sum / cols; var = sum_sq / cols - mean^2.
//   pass 2 — for each element: out = (x - mean) / sqrt(var + eps) * gamma + beta.
//
// All math inside the kernel uses FP32 (matches the FP32-accumulator
// convention for FP16/BF16). Output is converted back to dtype T on store.
//
// The two-pass reduction is fine for cols up to ~32K. A Welford-style
// online reduction is planned for a follow-up if profiling shows it
// matters at very wide cols (Wave 5+ optimization).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/layernorm.cuh"
#include "tensor/Dtype.hpp"

namespace tensorforge {

namespace {

constexpr int kLayerNormBlock = 256;

// fp16/bf16 -> float / float -> fp16/bf16 (mirror softmax/elementwise).
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
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_results[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_results[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (threadIdx.x < num_warps) ? warp_results[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}

template <typename T>
__global__ void layernorm_kernel(const T* __restrict__ in,
                                  const T* __restrict__ gamma,
                                  const T* __restrict__ beta,
                                  T* __restrict__ out,
                                  int64_t cols, float eps) {
    const T* row_in = in + (int64_t)blockIdx.x * cols;
    T* row_out = out + (int64_t)blockIdx.x * cols;

    // ---- Pass 1: sum and sum-of-squares ----
    float partial_sum = 0.0f;
    float partial_sumsq = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = to_float(row_in[c]);
        partial_sum += v;
        partial_sumsq += v * v;
    }
    float row_sum = block_reduce_sum(partial_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();
    float total = s_sum;
    float inv_cols = 1.0f / static_cast<float>(cols);
    float mean = total * inv_cols;

    float row_sumsq = block_reduce_sum(partial_sumsq);
    __shared__ float s_sumsq;
    if (threadIdx.x == 0) s_sumsq = row_sumsq;
    __syncthreads();
    float mean_sq = s_sumsq * inv_cols;
    float var = mean_sq - mean * mean;
    // Tiny negative variances (numerical noise) get clamped to 0.
    if (var < 0.0f) var = 0.0f;
    float inv_std = rsqrtf(var + eps);

    // ---- Pass 2: normalize + affine ----
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = to_float(row_in[c]);
        float g = to_float(gamma[c]);
        float b = to_float(beta[c]);
        float y = (v - mean) * inv_std * g + b;
        row_out[c] = from_float<T>(y);
    }
}

template <typename T>
void launch_layernorm_typed(const T* in, const T* gamma, const T* beta, T* out,
                           int64_t rows, int64_t cols, float eps,
                           cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 grid(static_cast<unsigned>(rows));
    dim3 block(static_cast<unsigned>(kLayerNormBlock));
    layernorm_kernel<T><<<grid, block, 0, stream>>>(in, gamma, beta, out,
                                                     cols, eps);
}

}  // namespace

void launch_layernorm(const void* in, const void* gamma, const void* beta,
                      void* out, int64_t rows, int64_t cols,
                      float eps, Dtype dtype, void* stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    switch (dtype) {
    case Dtype::Float32:
        launch_layernorm_typed(static_cast<const float*>(in),
                               static_cast<const float*>(gamma),
                               static_cast<const float*>(beta),
                               static_cast<float*>(out),
                               rows, cols, eps, cuda_stream);
        break;
    case Dtype::Float16:
        launch_layernorm_typed(static_cast<const __half*>(in),
                               static_cast<const __half*>(gamma),
                               static_cast<const __half*>(beta),
                               static_cast<__half*>(out),
                               rows, cols, eps, cuda_stream);
        break;
    case Dtype::BFloat16:
        launch_layernorm_typed(static_cast<const __nv_bfloat16*>(in),
                               static_cast<const __nv_bfloat16*>(gamma),
                               static_cast<const __nv_bfloat16*>(beta),
                               static_cast<__nv_bfloat16*>(out),
                               rows, cols, eps, cuda_stream);
        break;
    default:
        break;
    }
}

}  // namespace tensorforge
