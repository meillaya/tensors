// TensorForge - 2D matrix transpose CUDA kernel (Wave 6 / T28)
//
// Computes out[i, j] = in[j, i] for row-major 2D matrices of shape
// [M, N] -> [N, M]. Used by nn::Conv2d::backward to materialize the
// transposed weight matrix for grad_input computation. Templated on
// dtype (FP32 / FP16 / BF16) with no FP32 accumulator (the operation
// is exact).

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/transpose.cuh"
#include "tensor/Dtype.hpp"

namespace tensorforge {

namespace {

// launch_transpose_2d(in, out, M, N) treats `in` as a row-major [M, N]
// matrix and produces a row-major [N, M] matrix in `out`. Specifically
// out[i, j] = in[j, i] for i in [0, N), j in [0, M).
template <typename T>
__global__ void transpose_kernel(const T* __restrict__ in,
                                  T* __restrict__ out,
                                  int64_t M, int64_t N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // i in [0, N)
    int64_t j = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;  // j in [0, M)
    if (i >= N || j >= M) return;
    // out[i, j] = in[j, i]; out is row-major [N, M]
    out[i * M + j] = in[j * N + i];
}

template <typename T>
void launch_transpose_typed(const T* in, T* out, int64_t M, int64_t N,
                             cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    dim3 block(16, 16);
    // Grid covers out which has shape [N, M].
    dim3 grid(static_cast<unsigned int>((N + 15) / 16),
              static_cast<unsigned int>((M + 15) / 16));
    transpose_kernel<T><<<grid, block, 0, stream>>>(in, out, M, N);
}

}  // namespace

void launch_transpose_2d(const void* in, void* out, int64_t M, int64_t N,
                          Dtype dtype, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (dtype) {
        case Dtype::Float32:
            launch_transpose_typed<float>(static_cast<const float*>(in),
                                            static_cast<float*>(out), M, N, s);
            break;
        case Dtype::Float16:
            launch_transpose_typed<__half>(static_cast<const __half*>(in),
                                            static_cast<__half*>(out), M, N, s);
            break;
        case Dtype::BFloat16:
            launch_transpose_typed<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(in),
                static_cast<__nv_bfloat16*>(out), M, N, s);
            break;
        default:
            throw std::invalid_argument("launch_transpose_2d: unsupported dtype");
    }
}

}  // namespace tensorforge
