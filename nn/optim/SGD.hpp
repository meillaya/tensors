// TensorForge - nn/optim/SGD (T48)
//
// Vanilla SGD with optional momentum (Polyak / Sutskever-style):
//   if momentum == 0:     p <- p - lr * grad
//   if momentum  > 0:     v <- momentum * v + grad
//                         p <- p - lr * v
//
// The optimizer holds raw Tensor* (parameters_) plus a parallel
// `velocity_` buffer for momentum. zero_grad() resets accumulated
// gradients by memsetting the CPU-side storage of each parameter's
// `.grad()` to 0 — works because the autograd setup we link against
// always materialises gradients on CPU Float32.
//
// In-place parameter updates use `p = p - lr * g` (a fresh Tensor) so
// the autograd graph doesn't accumulate stale backward edges. Because
// the parameter Tensors are owned by Parameter objects stored in the
// nn::Module, we splice the new Tensor back through the parent's
// `data_` slot.

#pragma once

#include "tensor/Tensor.hpp"

#include <cstddef>
#include <vector>

namespace tensorforge::nn::optim {

class SGD {
public:
    SGD(std::vector<Tensor*> parameters, float lr, float momentum = 0.0f);

    // Update every parameter in-place using its current gradient.
    void step();

    // Reset every parameter's gradient to zero. Call once per step,
    // before backward().
    void zero_grad();

    // Number of parameters registered.
    [[nodiscard]] std::size_t size() const noexcept { return parameters_.size(); }

    // Convenience read-only accessor (mostly for tests / debugging).
    [[nodiscard]] const std::vector<Tensor*>& parameters() const noexcept {
        return parameters_;
    }

private:
    std::vector<Tensor*> parameters_;
    float lr_;
    float momentum_;
    std::vector<Tensor> velocity_;  // only used when momentum_ > 0
};

}  // namespace tensorforge::nn::optim
