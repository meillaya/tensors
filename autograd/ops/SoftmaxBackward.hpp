#pragma once

// TensorForge - SoftmaxBackward (Wave 7 / T34)
//
// Backward for y = softmax(x). The Jacobian-vector product for a
// softmax along a given dim is
//
//   grad_x[i] = y[i] * (grad_y[i] - sum_j(grad_y[j] * y[j]))
//
// which avoids materialising the full N x N Jacobian. Saves the softmax
// output `y` rather than the input `x` because the formula is most
// stable written in terms of y itself.

#include "autograd/Node.hpp"
#include "autograd/SavedTensor.hpp"
#include "tensor/Tensor.hpp"

#include <vector>

namespace tensorforge {

class SoftmaxBackward : public Node {
public:
    explicit SoftmaxBackward(const Tensor& y)
        : Node({}), y_(y) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        const Tensor& y = y_.unpack();
        Tensor gy = grads[0] * y;
        // Reduce along the last dim. The exact reduction shape is
        // handled by Tensor::sum(int dim, bool keepdim); here we sum
        // along the softmax dim with keepdim=true so broadcasting back
        // to y's shape works without reshaping.
        Tensor sum_gy = gy.sum(-1, true);
        return {y * (grads[0] - sum_gy)};
    }

private:
    SavedTensor y_;
};

} // namespace tensorforge
