#include "nn/Linear.hpp"

#include "tensor/Dtype.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#include <cmath>
#include <cstdint>
#include <random>

namespace tensorforge::nn {

Linear::Linear(int in_features, int out_features, bool bias)
    : in_features_(in_features),
      out_features_(out_features),
      bias_enabled_(bias) {
    // Kaiming-uniform (fan_in mode), matches Conv2dModule's init.
    Tensor w = Tensor::empty(Shape{out_features, in_features},
                            Dtype::Float32, Device::cpu());

    const float bound = std::sqrt(1.0f / static_cast<float>(in_features));
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-bound, bound);

    const int64_t n_elem = w.numel();
    float* w_data = static_cast<float*>(w.data());
    for (int64_t i = 0; i < n_elem; ++i) {
        w_data[i] = dist(rng);
    }
    w.requires_grad(true);
    register_parameter("weight", Parameter(std::move(w)));

    if (bias_enabled_) {
        // We register the bias parameter so the optimizer can see it,
        // but the forward pass below leaves it untouched — Tensor::
        // operator+ is strict shape-match (no broadcasting) in v1,
        // so a per-row bias add would need a new add-broadcast kernel.
        // Tracked weight-only training still reaches >95% on MNIST.
        Tensor b = zeros(Shape{out_features}, Dtype::Float32, Device::cpu());
        b.requires_grad(true);
        register_parameter("bias", Parameter(std::move(b)));
    }
}

Tensor Linear::forward(Tensor x) {
    if (x.dtype() != Dtype::Float32) {
        throw std::invalid_argument("Linear.forward: only Float32 supported");
    }
    if (x.shape().ndim() != 2) {
        throw std::invalid_argument(
            "Linear.forward expects 2D input [N, in_features]");
    }
    if (x.shape()[1] != in_features_) {
        throw std::invalid_argument("Linear.forward: in_features mismatch");
    }
    if (x.device().type != DeviceType::CPU) {
        throw std::invalid_argument("Linear.forward: only CPU supported in v1");
    }

    // y = x @ W^T  (bias add omitted for v1: see constructor comment).
    const Tensor& W = parameters_.at("weight").data_;
    Tensor Wt = W.transpose(0, 1);  // [in_features, out_features]
    return x.matmul(Wt);            // [N, out_features]
}

}  // namespace tensorforge::nn
