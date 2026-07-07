#pragma once

// TensorForge - TransposeBackward (T-FIX-3)
//
// Backward for y = x.transpose(dim0, dim1). Since transpose is its own
// inverse, the gradient w.r.t. x is simply the upstream gradient
// transposed along the same two dims:
//   d L / d x = (d L / d y).transpose(dim0, dim1)
//
// Only the swapped dims are saved; the forward input data is not needed
// for the backward pass.

#include "autograd/Node.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>
#include <vector>

namespace tensorforge {

class TransposeBackward : public Node {
public:
    TransposeBackward(int64_t dim0, int64_t dim1)
        : Node(std::vector<Edge>{}), dim0_(dim0), dim1_(dim1) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        return {grads[0].transpose(dim0_, dim1_)};
    }

private:
    int64_t dim0_;
    int64_t dim1_;
};

} // namespace tensorforge
