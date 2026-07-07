// TensorForge - reduce_sum_axis CUDA kernel (Wave 6 / T28)
//
// Reduces a 4D tensor [N, C, H, W] over dims 0, 2, 3 keeping dim 1, so
// the output has shape [C] and out[c] = sum over (n, h, w) of x[n, c, h, w].
// Used by nn::Conv2d::backward for grad_bias.
//
// One block per output channel; threads cooperate via shared-memory
// reduction. FP32 accumulator. Templated on input dtype (FP32 / FP16 /
// BF16) with FP32 accumulation so half-precision inputs do not lose
// precision during the reduction.

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/reduce_sum_axis.cuh"
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

// One block per channel. The block strides over (n, h, w) and
// accumulates partial sums in shared memory, then thread 0 writes the
// final scalar.
template <typename T>
__global__ void reduce_sum_axis_kernel(const T* __restrict__ x,
                                        T* __restrict__ out,
                                        int64_t N, int64_t C, int64_t H, int64_t W) {
    int64_t c = blockIdx.x;
    if (c >= C) return;
    int64_t HW = H * W;
    int64_t stride = (int64_t)blockDim.x * gridDim.y;
    int64_t start = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float smem[];

    float local = 0.0f;
    // Iterate n, then hw.
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t i = start; i < HW; i += stride) {
            int64_t offset = ((n * C + c) * H * W) + i;
            local += to_float(x[offset]);
        }
    }
    smem[threadIdx.x] = local;
    __syncthreads();

    // Standard tree reduction in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[c] = from_float<T>(smem[0]);
    }
}

}  // namespace

void launch_reduce_sum_axis_nchw(const void* x, void* out,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  Dtype dtype, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    constexpr int kBlock = 256;
    int gridY = static_cast<int>(((H * W) + kBlock - 1) / kBlock);
    if (gridY < 1) gridY = 1;
    dim3 grid(C, gridY);
    size_t smem_bytes = kBlock * sizeof(float);

    switch (dtype) {
        case Dtype::Float32:
            reduce_sum_axis_kernel<float><<<grid, kBlock, smem_bytes, s>>>(
                static_cast<const float*>(x), static_cast<float*>(out),
                N, C, H, W);
            break;
        case Dtype::Float16:
            reduce_sum_axis_kernel<__half><<<grid, kBlock, smem_bytes, s>>>(
                static_cast<const __half*>(x), static_cast<__half*>(out),
                N, C, H, W);
            break;
        case Dtype::BFloat16:
            reduce_sum_axis_kernel<__nv_bfloat16><<<grid, kBlock, smem_bytes, s>>>(
                static_cast<const __nv_bfloat16*>(x),
                static_cast<__nv_bfloat16*>(out),
                N, C, H, W);
            break;
        default:
            throw std::invalid_argument("launch_reduce_sum_axis_nchw: unsupported dtype");
    }
}

}  // namespace tensorforge
