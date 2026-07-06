#pragma once

#include "autograd/Node.hpp"
#include "tensor/Tensor.hpp"

#include <vector>

namespace tensorforge {

// AddBackward — backward for z = a + b.
// Given grad_z, returns {grad_z, grad_z} since dz/da = dz/db = 1.
// Used for testing the autograd engine; real ops come in T33+.
class AddBackward : public Node {
public:
    AddBackward() = default;

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
};

} // namespace tensorforge
