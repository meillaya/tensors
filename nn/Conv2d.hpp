// TensorForge — Conv2d module header (Wave 6 / T27)
//
// 2D convolution as a Module: weights of shape [Cout, Cin, kH, kW], bias
// of shape [Cout]. Forward path is the im2col + GEMM decomposition:
//
//   1. launch_im2col(input, col) — input [N, Cin, H, W] becomes a
//      column matrix col of shape [N, Cin*kH*kW, outH*outW].
//   2. Reshape weight to [Cout, Cin*kH*kW] and call launch_gemm_tiled_16x16.
//   3. Add bias (broadcast over Cout axis), reshape result back to
//      [N, Cout, outH, outW].
//
// Backward (T28) adds col2im for grad_input and two GEMM calls for
// grad_weight / grad_bias.
//
// All math runs on the device the input lives on; the launchers are
// stream-aware so the Module composes cleanly with CudaStream.

#pragma once

#include "nn/Module.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>

namespace tensorforge::nn {

class Conv2d : public Module {
public:
    // Construct with channel/kernel/padding parameters. Weights are
    // initialized to a constant (0.1f) so the test can pin expected
    // values; a proper init (Kaiming/He) is planned for a later wave.
    Conv2d(int64_t in_channels, int64_t out_channels,
           int64_t kernel_h, int64_t kernel_w,
           int64_t stride_h = 1, int64_t stride_w = 1,
           int64_t pad_h = 0, int64_t pad_w = 0,
           int64_t dilation_h = 1, int64_t dilation_w = 1,
           bool bias_enabled = true);

    // Forward: input shape [N, Cin, H, W] -> output [N, Cout, outH, outW].
    Tensor forward(Tensor input) override;

    // Accessors for the parameters (also reachable via named_parameters).
    [[nodiscard]] Parameter& weight() { return parameters_.at("weight"); }
    [[nodiscard]] Parameter& bias() { return parameters_.at("bias"); }

    // Hyperparameters used by forward; also needed by the backward kernel
    // when it lands in T28.
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

    // Output spatial dims for a given input H/W — public so callers (and
    // tests) can pre-allocate the output tensor.
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