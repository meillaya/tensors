#pragma once

// TensorForge - MatmulBackward (Wave 7 / T34)
//
// Backward for C = A @ B (2D matrices).
//   d L / d A = (d L / d C) @ B^T
//   d L / d B = A^T @ (d L / d C)
//
// Saves both inputs so the transposes can be computed lazily at
// backward time. The Tensor::transpose(int, int) instance method
// supports negative indexing so we can call .transpose(-1, -2).

#include "autograd/Node.hpp"
#include "autograd/SavedTensor.hpp"
#include "tensor/Tensor.hpp"

#include <vector>

namespace tensorforge {

class MatmulBackward : public Node {
public:
    MatmulBackward(const Tensor& a, const Tensor& b)
        : Node({}), a_(a), b_(b) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        const Tensor& a = a_.unpack();
        const Tensor& b = b_.unpack();
        Tensor b_t = b.transpose(-1, -2);
        Tensor a_t = a.transpose(-1, -2);
        Tensor ga = grads[0] * b_t;
        Tensor gb = a_t * grads[0];
        return {std::move(ga), std::move(gb)};
    }

private:
    SavedTensor a_;
    SavedTensor b_;
};

} // namespace tensorforge
