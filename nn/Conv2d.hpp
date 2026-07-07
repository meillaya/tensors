// TensorForge - Conv2d module header (Wave 6 / T27/T28)
//
// 2D convolution as a Module: weights of shape [Cout, Cin, kH, kW], bias
// of shape [Cout]. Forward path is the im2col + GEMM decomposition:
//
//   1. launch_im2col(input, col) - input [N, Cin, H, W] becomes a
//      column matrix col of shape [N, Cin*kH*kW, outH*outW].
//   2. Reshape weight to [Cout, Cin*kH*kW] and call launch_gemm_tiled_16x16.
//   3. Add bias (broadcast over Cout axis), reshape result back to
//      [N, Cout, outH, outW].
//
// Backward (T28) adds:
//   * grad_input  = col2im(transpose(W) @ grad_output_reshaped)
//   * grad_weight = grad_output_reshaped @ transpose(im2col(input))
//   * grad_bias   = sum(grad_output, dim=(0, 2, 3))
//
// All math runs on the device the input lives on; the launchers are
// stream-aware so the Module composes cleanly with CudaStream.

#pragma once

#include "nn/Module.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>

namespace tensorforge::nn {

// Output of Conv2d::backward. grad_input is the gradient with respect
// to the input, grad_weight / grad_bias are gradients with respect to
// the parameters.
struct Conv2dGrad {
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
};

class Conv2d : public Module {
public:
    Conv2d(int64_t in_channels, int64_t out_channels,
           int64_t kernel_h, int64_t kernel_w,
           int64_t stride_h = 1, int64_t stride_w = 1,
           int64_t pad_h = 0, int64_t pad_w = 0,
           int64_t dilation_h = 1, int64_t dilation_w = 1,
           bool bias_enabled = true);

    Tensor forward(Tensor input) override;

    // Backward pass. Computes the gradients of the loss w.r.t. the
    // input and the parameters (weight, bias if enabled).
    //
    // `grad_output` is the gradient of the loss w.r.t. the forward
    // output, of shape [N, Cout, outH, outW]. `input` is the original
    // forward input of shape [N, Cin, H, W].
    //
    // The returned Conv2dGrad carries:
    //   * grad_input of shape [N, Cin, H, W]
    //   * grad_weight of shape [Cout, Cin, kH, kW]
    //   * grad_bias of shape [Cout] (zero-filled tensor of size Cout
    //     if bias_enabled == false).
    Conv2dGrad backward(Tensor grad_output, Tensor input);

    [[nodiscard]] Parameter& weight() { return parameters_.at("weight"); }
    [[nodiscard]] Parameter& bias() { return parameters_.at("bias"); }

    int64_t in_channels() const noexcept { return in_channels_; }
    int64_t out_channels() const noexcept { return out_channels_; }
    int64_t kernel_h() const noexcept { return kernel_h_; }
    int64_t kernel_w() const noexcept { return kernel_w_; }
    int64_t stride_h() const noexcept { return stride_h_; }
    int64_t stride_w() const noexcept { return stride_w_; }
    int64_t pad_h() const noexcept { return pad_h_; }
    int64_t pad_w() const noexcept { return pad_w_; }
    int64_t dilation_h() const noexcept { return dilation_h_; }
    int64_t dilation_w() const noexcept { return dilation_w_; }
    bool bias_enabled() const noexcept { return bias_enabled_; }

    [[nodiscard]] int64_t out_h(int64_t H) const;
    [[nodiscard]] int64_t out_w(int64_t W) const;

private:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_h_;
    int64_t kernel_w_;
    int64_t stride_h_;
    int64_t stride_w_;
    int64_t pad_h_;
    int64_t pad_w_;
    int64_t dilation_h_;
    int64_t dilation_w_;
    bool bias_enabled_;
};

}  // namespace tensorforge::nn
