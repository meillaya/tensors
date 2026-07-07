// TensorForge — Conv2d module implementation (Wave 6 / T27)
//
// Forward path:
//   1. launch_im2col(input, col) — input [N, Cin, H, W] -> col
//      [N, Cin*kH*kW, outH*outW]
//   2. Reshape weight [Cout, Cin, kH, kW] to [Cout, Cin*kH*kW]
//   3. launch_gemm_tiled_16x16(W_2d, col) — produces [N, Cout, outH*outW]
//      (one GEMM per sample, laid out row-major as [N, Cout, M]).
//   4. launch_bias_add — adds bias[c] to each element of channel c,
//      in place, on the [N, Cout, M] output tensor.
//   5. Reshape result to [N, Cout, outH, outW].
//
// Backward (T28) will add:
//   * grad_input = col2im(transpose(W) @ grad_output_reshaped)
//   * grad_weight = grad_output_reshaped @ transpose(im2col(input))
//   * grad_bias = sum over batch + spatial of grad_output

#include "nn/Conv2d.hpp"

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/bias_add.cuh"
#include "cuda/kernels/gemm.cuh"
#include "cuda/kernels/im2col.cuh"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"
#include "tensor/shape_ops.hpp"

#include <cstdint>
#include <stdexcept>

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
    // Initialize weights to a small constant so the test expectations
    // are easy to compute by hand. Real init (Kaiming) lands later.
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
    // ---- shape checks ----
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

    // ---- 1. im2col ----
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

    // ---- 2. Reshape weight [Cout, Cin, kH, kW] -> [Cout, Cin*kH*kW] ----
    Parameter& w_param = weight();
    Tensor w2d = tensorforge::reshape(w_param.data_,
                                       {out_channels_, Cin * kernel_h_ * kernel_w_});

    // ---- 3. GEMM: output[N, Cout, outH*outW] = w2d @ col ----
    Tensor out_flat = tensorforge::Tensor::empty(
        {N, out_channels_, outH * outW}, dtype, input.device());

    int64_t K = Cin * kernel_h_ * kernel_w_;
    int64_t M_gemm = out_channels_;          // rows of w2d
    int64_t N_gemm = outH * outW;            // cols of col[n]
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

    // ---- 4. In-place bias add (per-channel broadcast) ----
    if (bias_enabled_) {
        Parameter& b_param = bias();
        tensorforge::launch_bias_add(out_flat.data(), b_param.data_.data(),
                                       N, out_channels_, outH * outW,
                                       dtype, stream);
    }

    // ---- 5. Reshape to [N, Cout, outH, outW] ----
    return tensorforge::reshape(out_flat, {N, out_channels_, outH, outW});
}

}  // namespace tensorforge::nn
