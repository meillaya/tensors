// TensorForge — softmax kernel implementation (Wave 4 / T20)
//
// Numerically stable softmax:
//
//   out[r, c] = exp(in[r, c] - row_max[r]) / sum_c' exp(in[r, c'] - row_max[r])
//
// Implementation strategy: ONE kernel, two passes per block (one block per
// row). Pass 1 reduces the row to its max using shared memory + warp
// shuffles. Pass 2 reads the max, computes exp(x - max), reduces to sum,
// then divides into the just-written exp values.
//
// All math is done in FP32 inside the kernel regardless of input dtype; the
// output is converted back to dtype T on store. This avoids the BF16/FP16
// exp() path which is not single-instruction on H100.
//
// Block size: 256 threads. cols may be larger — the per-block reduction
// strides by blockDim.x and uses shared memory to coordinate partials.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/softmax.cuh"
#include "tensor/Dtype.hpp"

namespace tensorforge {

namespace {

constexpr int kSoftmaxBlock = 256;

// fp16/bf16 -> float / float -> fp16/bf16 helpers (mirror elementwise.cu).
template <typename T>
struct CudaDtypeMap;

template <>
struct CudaDtypeMap<float> { static constexpr Dtype dtype = Dtype::Float32; };
template <>
struct CudaDtypeMap<__half> { static constexpr Dtype dtype = Dtype::Float16; };
template <>
struct CudaDtypeMap<__nv_bfloat16> {
    static constexpr Dtype dtype = Dtype::BFloat16;
};

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

// Warp-level max reduction via shuffle (returns result in lane 0).
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Warp-level sum reduction via shuffle.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level max reduction: each thread holds a partial max, threads in
// the same warp use warp_reduce_max, then one warp reduces across warps
// via shared memory.
__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float warp_results[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < num_warps) ? warp_results[lane] : -INFINITY;
        val = warp_reduce_max(val);
    }
    return val;  // only valid in warp 0; broadcasted via shared mem below.
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_results[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < num_warps) ? warp_results[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Softmax kernel: one block per row.
//
// Pass 1: each thread strides through its slice of the row finding the
//         max, then we block-reduce to a single scalar.
// Pass 2: each thread reads the broadcast max, computes exp(x - max),
//         block-reduces the sum, then divides by the sum and writes back.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void softmax_kernel(const T* __restrict__ in,
                                T* __restrict__ out,
                                int64_t cols) {
    const T* row_in = in + (int64_t)blockIdx.x * cols;
    T* row_out = out + (int64_t)blockIdx.x * cols;

    // Pass 1: per-thread max.
    float thread_max = -INFINITY;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        thread_max = fmaxf(thread_max, to_float(row_in[c]));
    }
    float row_max = block_reduce_max(thread_max);

    // Broadcast row_max to every thread via shared memory.
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = row_max;
    }
    __syncthreads();
    row_max = s_max;

    // Pass 2a: per-thread exp(x - max) and partial sum.
    float thread_sum = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = __expf(to_float(row_in[c]) - row_max);
        row_out[c] = from_float<T>(e);
        thread_sum += e;
    }
    float row_sum = block_reduce_sum(thread_sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_sum = row_sum;
    }
    __syncthreads();
    float inv = 1.0f / s_sum;

    // Pass 2b: divide in place.
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = to_float(row_out[c]) * inv;
        row_out[c] = from_float<T>(e);
    }
}

template <typename T>
void launch_softmax_typed(const T* in, T* out, int64_t rows, int64_t cols,
                          cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 grid(static_cast<unsigned>(rows));
    dim3 block(static_cast<unsigned>(kSoftmaxBlock));
    softmax_kernel<T><<<grid, block, 0, stream>>>(in, out, cols);
}

}  // namespace

void launch_softmax(const void* in, void* out, int64_t rows, int64_t cols,
                    Dtype dtype, void* stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    switch (dtype) {
    case Dtype::Float32:
        launch_softmax_typed(static_cast<const float*>(in),
                             static_cast<float*>(out), rows, cols, cuda_stream);
        break;
    case Dtype::Float16:
        launch_softmax_typed(static_cast<const __half*>(in),
                             static_cast<__half*>(out), rows, cols, cuda_stream);
        break;
    case Dtype::BFloat16:
        launch_softmax_typed(static_cast<const __nv_bfloat16*>(in),
                             static_cast<__nv_bfloat16*>(out), rows, cols, cuda_stream);
        break;
    default:
        break;
    }
}

}  // namespace tensorforge
