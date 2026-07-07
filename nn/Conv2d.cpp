// TensorForge - Conv2d module implementation (Wave 6 / T27/T28)
//
// Forward path (T27):
//   1. launch_im2col(input, col) - input [N, Cin, H, W] -> col
//      [N, Cin*kH*kW, outH*outW]
//   2. Reshape weight [Cout, Cin, kH, kW] to [Cout, Cin*kH*kW]
//   3. launch_gemm_tiled_16x16(W_2d, col) - produces [N, Cout, outH*outW]
//   4. launch_bias_add - adds bias[c] to each element of channel c
//   5. Reshape result to [N, Cout, outH, outW].
//
// Backward path (T28):
//   grad_input  = col2im(transpose(W) @ grad_output_reshaped)
//   grad_weight = sum_n ( grad_output[n] @ col[n]^T )
//   grad_bias   = sum_{n,h,w} grad_output[n, c, h, w]

#include "nn/Conv2d.hpp"

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/bias_add.cuh"
#include "cuda/kernels/col2im.cuh"
#include "cuda/kernels/gemm.cuh"
#include "cuda/kernels/im2col.cuh"
#include "cuda/kernels/reduce_sum_axis.cuh"
#include "cuda/kernels/transpose.cuh"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"
#include "tensor/shape_ops.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace tensorforge::nn {

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels,
                int64_t kernel_h, int64_t kernel_w,
                int64_t stride_h, int64_t stride_w,
                int64_t pad_h, int64_t pad_w,
                int64_t dilation_h, int64_t dilation_w,
                bool bias_enabled)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      pad_h_(pad_h),
      pad_w_(pad_w),
      dilation_h_(dilation_h),
      dilation_w_(dilation_w),
      bias_enabled_(bias_enabled) {
    Tensor w = full({out_channels, in_channels, kernel_h, kernel_w},
                     0.1f, Dtype::Float32, Device::cuda(0));
    w.requires_grad(true);
    register_parameter("weight", Parameter(std::move(w)));

    if (bias_enabled) {
        Tensor b = full({out_channels}, 0.05f, Dtype::Float32, Device::cuda(0));
        b.requires_grad(true);
        register_parameter("bias", Parameter(std::move(b)));
    }
}

int64_t Conv2d::out_h(int64_t H) const {
    return (H + 2 * pad_h_ - dilation_h_ * (kernel_h_ - 1) - 1) / stride_h_ + 1;
}

int64_t Conv2d::out_w(int64_t W) const {
    return (W + 2 * pad_w_ - dilation_w_ * (kernel_w_ - 1) - 1) / stride_w_ + 1;
}

Tensor Conv2d::forward(Tensor input) {
    const auto& in_shape = input.shape();
    if (in_shape.ndim() != 4) {
        throw std::invalid_argument("Conv2d.forward expects [N, Cin, H, W]");
    }
    int64_t N  = in_shape[0];
    int64_t Cin = in_shape[1];
    int64_t H = in_shape[2];
    int64_t W = in_shape[3];
    if (Cin != in_channels_) {
        throw std::invalid_argument("Conv2d.forward: Cin mismatch");
    }

    int64_t outH = out_h(H);
    int64_t outW = out_w(W);

    auto& ctx = tensorforge::DeviceContext::current();
    void* stream = ctx.current_stream;
    Dtype dtype = input.dtype();

    Tensor col = tensorforge::Tensor::empty(
        {N, Cin * kernel_h_ * kernel_w_, outH * outW},
        dtype, input.device());

    tensorforge::launch_im2col(input.data(), col.data(),
                                N, Cin, H, W,
                                kernel_h_, kernel_w_,
                                stride_h_, stride_w_,
                                pad_h_, pad_w_,
                                dilation_h_, dilation_w_,
                                dtype, stream);

    Parameter& w_param = weight();
    Tensor w2d = tensorforge::reshape(w_param.data_,
                                       {out_channels_, Cin * kernel_h_ * kernel_w_});

    Tensor out_flat = tensorforge::Tensor::empty(
        {N, out_channels_, outH * outW}, dtype, input.device());

    int64_t K = Cin * kernel_h_ * kernel_w_;
    int64_t M_gemm = out_channels_;
    int64_t N_gemm = outH * outW;
    for (int64_t n = 0; n < N; ++n) {
        const char* col_base = static_cast<const char*>(col.data())
            + n * (K * N_gemm) * dtype_size(dtype);
        char* out_base = static_cast<char*>(out_flat.data())
            + n * (M_gemm * N_gemm) * dtype_size(dtype);
        tensorforge::launch_gemm_tiled_16x16(
            w2d.data(), col_base, out_base,
            M_gemm, N_gemm, K,
            dtype, stream);
    }

    if (bias_enabled_) {
        Parameter& b_param = bias();
        tensorforge::launch_bias_add(out_flat.data(), b_param.data_.data(),
                                       N, out_channels_, outH * outW,
                                       dtype, stream);
    }

    return tensorforge::reshape(out_flat, {N, out_channels_, outH, outW});
}

Conv2dGrad Conv2d::backward(Tensor grad_output, Tensor input) {
    const auto& go_shape = grad_output.shape();
    if (go_shape.ndim() != 4) {
        throw std::invalid_argument("Conv2d.backward: grad_output must be [N, Cout, outH, outW]");
    }
    const auto& in_shape = input.shape();
    if (in_shape.ndim() != 4) {
        throw std::invalid_argument("Conv2d.backward: input must be [N, Cin, H, W]");
    }
    int64_t N = go_shape[0];
    int64_t Cout = go_shape[1];
    int64_t outH = go_shape[2];
    int64_t outW = go_shape[3];
    int64_t Cin = in_shape[1];
    int64_t H = in_shape[2];
    int64_t W = in_shape[3];
    if (Cout != out_channels_ || Cin != in_channels_) {
        throw std::invalid_argument("Conv2d.backward: channel mismatch");
    }

    auto& ctx = tensorforge::DeviceContext::current();
    void* stream = ctx.current_stream;
    Dtype dtype = grad_output.dtype();
    Device device = grad_output.device();
    int64_t K = Cin * kernel_h_ * kernel_w_;
    int64_t M_gemm = outH * outW;

    // ---- 1. im2col(input) -> col of shape [N, K, M_gemm] ----
    Tensor col = tensorforge::Tensor::empty(
        {N, K, M_gemm}, dtype, device);
    tensorforge::launch_im2col(input.data(), col.data(),
                                N, Cin, H, W,
                                kernel_h_, kernel_w_,
                                stride_h_, stride_w_,
                                pad_h_, pad_w_,
                                dilation_h_, dilation_w_,
                                dtype, stream);

    // ---- 2. grad_input = col2im(W^T @ grad_output_reshaped) ----
    // W is [Cout, K]; W^T in row-major is [K, Cout]. Materialise it.
    Parameter& w_param = weight();
    Tensor w2d = tensorforge::reshape(w_param.data_, {out_channels_, K});
    Tensor w_T = tensorforge::Tensor::empty({K, out_channels_}, dtype, device);
    tensorforge::launch_transpose_2d(w2d.data(), w_T.data(),
                                      out_channels_, K, dtype, stream);

    Tensor col_grad = tensorforge::Tensor::empty({N, K, M_gemm}, dtype, device);
    for (int64_t n = 0; n < N; ++n) {
        const char* go_base = static_cast<const char*>(grad_output.data())
            + n * (Cout * M_gemm) * dtype_size(dtype);
        char* cg_base = static_cast<char*>(col_grad.data())
            + n * (K * M_gemm) * dtype_size(dtype);
        // col_grad[n] = W^T @ grad_output[n]
        //   W^T is [K, Cout], grad_output[n] is [Cout, M_gemm], result is [K, M_gemm]
        tensorforge::launch_gemm_tiled_16x16(
            w_T.data(), go_base, cg_base,
            K, M_gemm, Cout,
            dtype, stream);
    }

    Tensor grad_input = tensorforge::zeros({N, Cin, H, W}, dtype, device);
    tensorforge::launch_col2im(col_grad.data(), grad_input.data(),
                                N, Cin, H, W,
                                kernel_h_, kernel_w_,
                                stride_h_, stride_w_,
                                pad_h_, pad_w_,
                                dilation_h_, dilation_w_,
                                outH, outW,
                                dtype, stream);

    // ---- 3. grad_weight = sum_n (grad_output[n] @ col[n]^T) ----
    // Slow path: per-sample GEMM, host-side accumulation. A future wave
    // adds an in-place GEMM-accumulate kernel.
    Tensor grad_weight = tensorforge::zeros({Cout, K}, dtype, device);
    Tensor gw_acc = tensorforge::Tensor::empty({Cout, K}, dtype, device);
    for (int64_t n = 0; n < N; ++n) {
        const char* go_base = static_cast<const char*>(grad_output.data())
            + n * (Cout * M_gemm) * dtype_size(dtype);
        const char* col_base = static_cast<const char*>(col.data())
            + n * (K * M_gemm) * dtype_size(dtype);
        // gw_acc = grad_output[n] @ col[n]^T
        //   grad_output[n] is [Cout, M_gemm], col[n]^T is [M_gemm, K]
        // Materialise col[n]^T of shape [M_gemm, K] then GEMM.
        Tensor col_n_T = tensorforge::Tensor::empty({M_gemm, K}, dtype, device);
        // col[n] is [K, M_gemm] in row-major; we want col_n_T of shape
        // [M_gemm, K] in row-major. launch_transpose_2d(in, out, M, N)
        // treats `in` as row-major [M, N] and produces row-major [N, M].
        // So pass M=K, N=M_gemm to get out of shape [M_gemm, K].
        tensorforge::launch_transpose_2d(col_base, col_n_T.data(),
                                          K, M_gemm, dtype, stream);
        tensorforge::launch_gemm_tiled_16x16(
            go_base, col_n_T.data(), gw_acc.data(),
            Cout, K, M_gemm,
            dtype, stream);
        // Synchronise and accumulate on host (slow, but correct).
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
        int64_t bytes = Cout * K * dtype_size(dtype);
        std::vector<float> gw_h(static_cast<size_t>(Cout * K));
        std::vector<float> acc_h(static_cast<size_t>(Cout * K));
        cudaMemcpy(gw_h.data(), grad_weight.data(), bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(acc_h.data(), gw_acc.data(), bytes, cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < Cout * K; ++i) gw_h[i] += acc_h[i];
        cudaMemcpy(grad_weight.data(), gw_h.data(), bytes, cudaMemcpyHostToDevice);
    }
    Tensor grad_weight_full = tensorforge::reshape(grad_weight,
                                                    {Cout, Cin, kernel_h_, kernel_w_});

    // ---- 4. grad_bias = reduce_sum_axis(grad_output) ----
    Tensor grad_bias = tensorforge::Tensor::empty({Cout}, dtype, device);
    if (bias_enabled_) {
        tensorforge::launch_reduce_sum_axis_nchw(
            grad_output.data(), grad_bias.data(),
            N, Cout, outH, outW, dtype, stream);
    } else {
        int64_t bytes = Cout * dtype_size(dtype);
        cudaMemsetAsync(grad_bias.data(), 0, bytes,
                         reinterpret_cast<cudaStream_t>(stream));
    }

    return Conv2dGrad{grad_input, grad_weight_full, grad_bias};
}

}  // namespace tensorforge::nn
