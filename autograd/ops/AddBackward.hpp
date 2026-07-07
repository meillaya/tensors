#pragma once

// TensorForge - AddBackward (Wave 7 / T33)
//
// Backward for z = a + b. Given grad_z, returns {grad_z, grad_z} since
// dz/da = dz/db = 1. Header-only since the implementation is trivial.

#include "autograd/Node.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>
#include <vector>

namespace tensorforge {

class AddBackward : public Node {
public:
    AddBackward() : Node({}) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        // z = a + b  ->  dz/da = 1, dz/db = 1
        // Pass-through: both input gradients equal the output gradient.
        return {grads[0], grads[0]};
    }
};

} // namespace tensorforge
