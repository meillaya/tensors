// TensorForge — im2col kernel implementation (Wave 5 / T26)
//
// Maps a [N, C, H, W] 4D input into a [N, C*kH*kW, outH*outW] 3D column
// matrix used by GEMM-based convolution. See im2col.cuh for shape
// definitions.
//
// One thread per output element. Output dims are linearized as
//     n * (C*kH*kW * outH*outW) + (c*kH*kW + ki*kW + kj) * (outH*outW) + (oh*outW + ow)
// Each thread:
//   1. decodes (n, c, ki, kj, oh, ow) from its flat index,
//   2. computes the source (ih, iw) = (oh*stride_h + ki*dilation_h - pad_h,
//                                    ow*stride_w + kj*dilation_w - pad_w),
//   3. if (ih, iw) is in [0, H) x [0, W), writes input[n, c, ih, iw],
//      else writes 0.
//
// All math in FP32 (matches the FP32-accumulator convention). Dtype
// templated on FP32 / FP16 / BF16.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/im2col.cuh"
#include "tensor/Dtype.hpp"

namespace tensorforge {

namespace {

// FP16 / BF16 <-> FP32 conversions. Same pattern as softmax/layernorm/gemm.
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

template <typename T>
__global__ void im2col_kernel(const T* __restrict__ input,
                               T* __restrict__ col,
                               int64_t N, int64_t C, int64_t H, int64_t W,
                               int64_t kH, int64_t kW,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w,
                               int64_t dilation_h, int64_t dilation_w,
                               int64_t outH, int64_t outW) {
    // Flat output index → (n, row, col_spatial)
    int64_t total_per_n = C * kH * kW * outH * outW;
    int64_t total = N * total_per_n;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t n = idx / total_per_n;
    int64_t rem = idx - n * total_per_n;

    int64_t spatial = outH * outW;
    int64_t row = rem / spatial;
    int64_t out_idx = rem - row * spatial;

    // Decode row → (c, ki, kj).
    int64_t kHkW = kH * kW;
    int64_t c = row / kHkW;
    int64_t rk = row - c * kHkW;
    int64_t ki = rk / kW;
    int64_t kj = rk - ki * kW;

    // Decode out_idx → (oh, ow).
    int64_t oh = out_idx / outW;
    int64_t ow = out_idx - oh * outW;

    int64_t ih = oh * stride_h + ki * dilation_h - pad_h;
    int64_t iw = ow * stride_w + kj * dilation_w - pad_w;

    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        int64_t in_offset = ((n * C + c) * H + ih) * W + iw;
        v = to_float(input[in_offset]);
    }
    col[idx] = from_float<T>(v);
}

template <typename T>
void launch_im2col_typed(const T* input, T* col,
                          int64_t N, int64_t C, int64_t H, int64_t W,
                          int64_t kH, int64_t kW,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w,
                          int64_t dilation_h, int64_t dilation_w,
                          cudaStream_t stream) {
    int64_t outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    int64_t total = N * C * kH * kW * outH * outW;
    if (total <= 0) return;
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    if (grid < 1) grid = 1;
    im2col_kernel<T><<<grid, block, 0, stream>>>(input, col,
        N, C, H, W, kH, kW,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, outH, outW);
}

}  // namespace

void launch_im2col(const void* input, void* col,
                   int64_t N, int64_t C, int64_t H, int64_t W,
                   int64_t kH, int64_t kW,
                   int64_t stride_h, int64_t stride_w,
                   int64_t pad_h, int64_t pad_w,
                   int64_t dilation_h, int64_t dilation_w,
                   Dtype dtype, void* stream) {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    switch (dtype) {
    case Dtype::Float32:
        launch_im2col_typed(static_cast<const float*>(input),
                            static_cast<float*>(col),
                            N, C, H, W, kH, kW,
                            stride_h, stride_w, pad_h, pad_w,
                            dilation_h, dilation_w, cuda_stream);
        break;
    case Dtype::Float16:
        launch_im2col_typed(static_cast<const __half*>(input),
                            static_cast<__half*>(col),
                            N, C, H, W, kH, kW,
                            stride_h, stride_w, pad_h, pad_w,
                            dilation_h, dilation_w, cuda_stream);
        break;
    case Dtype::BFloat16:
        launch_im2col_typed(static_cast<const __nv_bfloat16*>(input),
                            static_cast<__nv_bfloat16*>(col),
                            N, C, H, W, kH, kW,
                            stride_h, stride_w, pad_h, pad_w,
                            dilation_h, dilation_w, cuda_stream);
        break;
    default:
        break;
    }
}

}  // namespace tensorforge