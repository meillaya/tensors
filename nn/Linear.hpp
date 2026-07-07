// TensorForge - nn/Linear (T47)
//
// Fully-connected / linear layer: y = x @ W^T + b.
//
// Weight: [out_features, in_features] Float32, Kaiming-uniform init
// (bound = sqrt(1 / in_features)), matches nn.Conv2dModule's fan_in scheme.
// Bias: [out_features] Float32, initialised to zero. Both parameters
// require_grad.
//
// Forward builds `x @ W^T + b` out of pre-existing wirers (Tensor::matmul
// installs MatmulBackward; Tensor::operator+ installs AddBackward) so
// autograd lights up end-to-end without any new backward kernels.
//
// CPU-only in v1 — paths in Tensor::matmul and Tensor::operator+ require
// CPU Float32 2D operands.

#pragma once

#include "nn/Module.hpp"
#include "tensor/Tensor.hpp"

namespace tensorforge::nn {

class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool bias = true);

    [[nodiscard]] Tensor forward(Tensor x) override;

    [[nodiscard]] int in_features() const noexcept { return in_features_; }
    [[nodiscard]] int out_features() const noexcept { return out_features_; }

private:
    int in_features_;
    int out_features_;
    bool bias_enabled_;
};

}  // namespace tensorforge::nn
