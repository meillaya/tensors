#pragma once

// TensorForge - ReluBackward (Wave 7 / T34)
//
// Backward for y = max(x, 0). d y / d x = 1 if x > 0 else 0. The
// saved input is unpacked lazily; the wire-up code attaches this Node
// to the output of Tensor::relu() when requires_grad is true.

#include "autograd/Node.hpp"
#include "autograd/SavedTensor.hpp"
#include "tensor/Tensor.hpp"

#include <vector>

namespace tensorforge {

class ReluBackward : public Node {
public:
    explicit ReluBackward(const Tensor& input)
        : Node(std::vector<Edge>{}), input_(input) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        const Tensor& x = input_.unpack();
        // mask = (x > 0) cast to x's dtype and device.
        Tensor mask = Tensor::empty(x.shape(), x.dtype(), x.device());
        const float* xp = static_cast<const float*>(x.data());
        float* mp = static_cast<float*>(mask.data());
        for (int64_t i = 0; i < x.numel(); ++i) {
            mp[i] = xp[i] > 0.0f ? 1.0f : 0.0f;
        }
        return {grads[0] * mask};
    }

private:
    SavedTensor input_;
};

} // namespace tensorforge
