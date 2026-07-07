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
        : Node(std::vector<Edge>{}), y_(y) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        const Tensor& y = y_.unpack();
        Tensor gy = grads[0] * y;
        // Reduce along the last dim (no keepdim — yields shape with the
        // last axis dropped). Then materialise the per-row scalar back
        // to a full tensor of y's shape so the subtraction below has
        // matching shapes. The Tensor API doesn't have broadcast yet,
        // so we expand explicitly.
        Tensor sum_gy = gy.sum(-1, false);
        int64_t outer = y.numel() / y.shape()[y.shape().ndim() - 1];
        int64_t inner = y.shape()[y.shape().ndim() - 1];
        Tensor sum_gy_full = Tensor::empty(y.shape(), y.dtype(), y.device());
        const float* sp = static_cast<const float*>(sum_gy.data());
        float* sfp = static_cast<float*>(sum_gy_full.data());
        for (int64_t r = 0; r < outer; ++r) {
            for (int64_t j = 0; j < inner; ++j) {
                sfp[r * inner + j] = sp[r];
            }
        }
        // grad_y = y * (grad_out - sum_gy)
        Tensor diff = Tensor::empty(y.shape(), y.dtype(), y.device());
        const float* gp = static_cast<const float*>(grads[0].data());
        const float* yp = static_cast<const float*>(y.data());
        float* dp = static_cast<float*>(diff.data());
        for (int64_t i = 0; i < y.numel(); ++i) {
            dp[i] = gp[i] - sfp[i];
        }
        return {y * diff};
    }

private:
    SavedTensor y_;
};

} // namespace tensorforge
