#pragma once

#include "autograd/Node.hpp"
#include "autograd/SavedTensor.hpp"
#include "tensor/Tensor.hpp"

#include <vector>

namespace tensorforge {

// MulBackward — backward for z = a * b.
// Given grad_z, returns {grad_z * b, a * grad_z} since dz/da = b, dz/db = a.
// Saves both inputs via SavedTensor so the unpack is consistent with
// in-place mutation detection.
class MulBackward : public Node {
public:
    MulBackward(const Tensor& a, const Tensor& b)
        : Node(std::vector<Edge>{}), a_(a), b_(b) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        const Tensor& a = a_.unpack();
        const Tensor& b = b_.unpack();
        Tensor ga = grads[0] * b;
        Tensor gb = a * grads[0];
        return {std::move(ga), std::move(gb)};
    }

private:
    SavedTensor a_;
    SavedTensor b_;
};

} // namespace tensorforge