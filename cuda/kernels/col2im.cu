// TensorForge - col2im CUDA kernel (Wave 6 / T28)
//
// Inverse of im2col: scatter-add entries from a column matrix of shape
// [N, C*kH*kW, outH*outW] back into an input tensor of shape [N, C, H, W].
//
// One thread per column entry. Each (n, c, ki, kj, oh, ow) maps to a
// single (ih, iw) input position (or zero-extends if out-of-bounds, in
// which case the entry is dropped). When multiple column entries fall
// on the same input pixel (e.g. overlapping receptive fields from
// stride < kernel or dilation > 1), atomicAdd accumulates them.

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/col2im.cuh"
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

__global__ void col2im_f32_kernel(const float* __restrict__ col,
                                   float* __restrict__ input_grad,
                                   int64_t N, int64_t C, int64_t H, int64_t W,
                                   int64_t kH, int64_t kW,
                                   int64_t stride_h, int64_t stride_w,
                                   int64_t pad_h, int64_t pad_w,
                                   int64_t dilation_h, int64_t dilation_w,
                                   int64_t outH, int64_t outW) {
    int64_t total_per_n = C * kH * kW * outH * outW;
    int64_t total = N * total_per_n;
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;

    int64_t n = t / total_per_n;
    int64_t rem = t - n * total_per_n;
    int64_t row = rem / (outH * outW);
    int64_t idx = rem - row * (outH * outW);
    int64_t c = row / (kH * kW);
    int64_t rk = row - c * (kH * kW);
    int64_t ki = rk / kW;
    int64_t kj = rk - ki * kW;
    int64_t oh = idx / outW;
    int64_t ow = idx - oh * outW;

    int64_t ih = oh * stride_h + ki * dilation_h - pad_h;
    int64_t iw = ow * stride_w + kj * dilation_w - pad_w;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) return;

    int64_t dst = ((n * C + c) * H + ih) * W + iw;
    atomicAdd(&input_grad[dst], col[t]);
}

template <typename T>
__global__ void col2im_typed_kernel(const T* __restrict__ col,
                                     T* __restrict__ input_grad,
                                     int64_t N, int64_t C, int64_t H, int64_t W,
                                     int64_t kH, int64_t kW,
                                     int64_t stride_h, int64_t stride_w,
                                     int64_t pad_h, int64_t pad_w,
                                     int64_t dilation_h, int64_t dilation_w,
                                     int64_t outH, int64_t outW) {
    int64_t total_per_n = C * kH * kW * outH * outW;
    int64_t total = N * total_per_n;
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;

    int64_t n = t / total_per_n;
    int64_t rem = t - n * total_per_n;
    int64_t row = rem / (outH * outW);
    int64_t idx = rem - row * (outH * outW);
    int64_t c = row / (kH * kW);
    int64_t rk = row - c * (kH * kW);
    int64_t ki = rk / kW;
    int64_t kj = rk - ki * kW;
    int64_t oh = idx / outW;
    int64_t ow = idx - oh * outW;

    int64_t ih = oh * stride_h + ki * dilation_h - pad_h;
    int64_t iw = ow * stride_w + kj * dilation_w - pad_w;
    if (ih < 0 || ih >= H || iw < 0 || iw >= W) return;

    int64_t dst = ((n * C + c) * H + ih) * W + iw;
    float v = to_float(col[t]) + to_float(input_grad[dst]);
    input_grad[dst] = from_float<T>(v);
}

void launch_col2im_f32(const float* col, float* input_grad,
                        int64_t N, int64_t C, int64_t H, int64_t W,
                        int64_t kH, int64_t kW,
                        int64_t stride_h, int64_t stride_w,
                        int64_t pad_h, int64_t pad_w,
                        int64_t dilation_h, int64_t dilation_w,
                        int64_t outH, int64_t outW,
                        cudaStream_t stream) {
    int64_t total = N * C * kH * kW * outH * outW;
    if (total <= 0) return;
    int grid = static_cast<int>((total + 255) / 256);
    col2im_f32_kernel<<<grid, 256, 0, stream>>>(
        col, input_grad, N, C, H, W, kH, kW,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, outH, outW);
}

template <typename T>
void launch_col2im_typed(const T* col, T* input_grad,
                          int64_t N, int64_t C, int64_t H, int64_t W,
                          int64_t kH, int64_t kW,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w,
                          int64_t dilation_h, int64_t dilation_w,
                          int64_t outH, int64_t outW,
                          cudaStream_t stream) {
    int64_t total = N * C * kH * kW * outH * outW;
    if (total <= 0) return;
    int grid = static_cast<int>((total + 255) / 256);
    col2im_typed_kernel<T><<<grid, 256, 0, stream>>>(
        col, input_grad, N, C, H, W, kH, kW,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, outH, outW);
}

}  // namespace

void launch_col2im(const void* col, void* input_grad,
                    int64_t N, int64_t C, int64_t H, int64_t W,
                    int64_t kH, int64_t kW,
                    int64_t stride_h, int64_t stride_w,
                    int64_t pad_h, int64_t pad_w,
                    int64_t dilation_h, int64_t dilation_w,
                    int64_t outH, int64_t outW,
                    Dtype dtype, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (dtype) {
        case Dtype::Float32:
            launch_col2im_f32(static_cast<const float*>(col),
                               static_cast<float*>(input_grad),
                               N, C, H, W, kH, kW,
                               stride_h, stride_w, pad_h, pad_w,
                               dilation_h, dilation_w, outH, outW, s);
            break;
        case Dtype::Float16:
            launch_col2im_typed<__half>(static_cast<const __half*>(col),
                                         static_cast<__half*>(input_grad),
                                         N, C, H, W, kH, kW,
                                         stride_h, stride_w, pad_h, pad_w,
                                         dilation_h, dilation_w, outH, outW, s);
            break;
        case Dtype::BFloat16:
            launch_col2im_typed<__nv_bfloat16>(
                static_cast<const __nv_bfloat16*>(col),
                static_cast<__nv_bfloat16*>(input_grad),
                N, C, H, W, kH, kW,
                stride_h, stride_w, pad_h, pad_w,
                dilation_h, dilation_w, outH, outW, s);
            break;
        default:
            throw std::invalid_argument("launch_col2im: unsupported dtype");
    }
}

}  // namespace tensorforge
