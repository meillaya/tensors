#pragma once

// TensorForge - LayerNormBackward (Wave 7 / T34)
//
// Backward for y = (x - mean) / sqrt(var + eps) * gamma + beta.
//
// Full backward for y against x, gamma, beta is moderately involved
// (mean/var backprop through the reduction axes). For T34 we ship a
// simplified Node that returns just grad_x = grad_y * gamma and
// stub-grads (zeros of the right shape) for gamma/beta. The real
// derivation lands in a follow-up wave once LayerNorm is used inside
// a training loop end-to-end.
//
// Callers should only rely on grad_x for now. gamma.grad / beta.grad
// are placeholders so the node returns the right number of gradients
// for the wiring.

#include "autograd/Node.hpp"
#include "autograd/SavedTensor.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#include <vector>

namespace tensorforge {

class LayerNormBackward : public Node {
public:
    LayerNormBackward(const Tensor& x, const Tensor& mean,
                      const Tensor& rstd, const Tensor& gamma)
        : Node(std::vector<Edge>{}), x_(x), mean_(mean), rstd_(rstd), gamma_(gamma) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        (void)x_.unpack();
        (void)mean_.unpack();
        (void)rstd_.unpack();
        const Tensor& gamma = gamma_.unpack();

        const Tensor& grad_y = grads[0];
        // Simplified grad_x: grad_y * gamma (no mean/var correction).
        // gamma is 1D [cols]; grad_y is 2D [rows, cols]. Broadcast
        // gamma along the row axis by manually expanding, since the
        // Tensor API has no broadcast yet.
        Tensor grad_x = Tensor::empty(grad_y.shape(), grad_y.dtype(), grad_y.device());
        int64_t rows = grad_y.shape()[0];
        int64_t cols = grad_y.shape()[1];
        const float* gyp = static_cast<const float*>(grad_y.data());
        const float* gp = static_cast<const float*>(gamma.data());
        float* gxp = static_cast<float*>(grad_x.data());
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t j = 0; j < cols; ++j) {
                gxp[r * cols + j] = gyp[r * cols + j] * gp[j];
            }
        }

        Tensor grad_gamma = zeros(gamma.shape(), gamma.dtype(), gamma.device());
        Tensor grad_beta = zeros(gamma.shape(), gamma.dtype(), gamma.device());

        return {std::move(grad_x), std::move(grad_gamma), std::move(grad_beta)};
    }

private:
    SavedTensor x_;
    SavedTensor mean_;
    SavedTensor rstd_;
    SavedTensor gamma_;
};

} // namespace tensorforge
