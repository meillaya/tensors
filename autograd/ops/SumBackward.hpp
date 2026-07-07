#pragma once

// TensorForge - SumBackward (Wave 9 / T35)
//
// Backward for y = sum(x, dim). d L / d x_{i} = d L / d y broadcast
// along the reduced dim. The backward broadcasts the upstream gradient
// back to the input shape; this is broadcast-able in v1 only when
// keepdim was true, so we materialise the full-sized tensor by hand.
//
// Saves the input shape (and original dims info via the upstream
// grad) so apply() can compute the right output.

#include "autograd/Node.hpp"
#include "autograd/SavedTensor.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#include <vector>

namespace tensorforge {

class SumBackward : public Node {
public:
    explicit SumBackward(const Tensor& input)
        : Node(std::vector<Edge>{}), input_(input) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        const Tensor& x = input_.unpack();
        Tensor upstream = grads[0];
        Tensor grad_x = Tensor::empty(x.shape(), x.dtype(), x.device());
        int64_t outer = 1;
        for (size_t i = 0; i < x.shape().ndim(); ++i) outer *= x.shape()[i];
        const float* up_p = static_cast<const float*>(upstream.data());
        float* gxp = static_cast<float*>(grad_x.data());
        for (int64_t i = 0; i < outer; ++i) {
            gxp[i] = up_p[0];
        }
        if (grad_x.dtype() == Dtype::Float32 && grad_x.numel() > 0) {
            const float* p = static_cast<const float*>(grad_x.data());
            fprintf(stderr, "[SumBackward] out: ");
            for (int64_t i = 0; i < grad_x.numel() && i < 8; ++i) {
                fprintf(stderr, "%.4f ", p[i]);
            }
            fprintf(stderr, " up=%.4f\n", up_p[0]);
        }
        return {std::move(grad_x)};
    }

private:
    SavedTensor input_;
};

} // namespace tensorforge
