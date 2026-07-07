#pragma once

// TensorForge - LogSoftmaxBackward
//
// Backward for y = log_softmax(x) along the last dim.
//
//   dL/dx_i = dL/dy_i - softmax(x)_i * sum_j(dL/dy_j)
//
// We save the log_softmax output `y`; the softmax needed for the
// backward is recovered as exp(y) elementwise.

#include "autograd/Node.hpp"
#include "autograd/SavedTensor.hpp"
#include "tensor/Tensor.hpp"

#include <cmath>
#include <vector>

namespace tensorforge {

class LogSoftmaxBackward : public Node {
public:
    explicit LogSoftmaxBackward(const Tensor& y)
        : Node(std::vector<Edge>{}), y_(y) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        const Tensor& y = y_.unpack();
        const Tensor& gy = grads[0];

        const int64_t ndim = y.shape().ndim();
        const int64_t last = y.shape()[ndim - 1];
        const int64_t outer = y.numel() / last;

        Tensor softmax_y = Tensor::empty(y.shape(), y.dtype(), y.device());
        const float* yp = static_cast<const float*>(y.data());
        float* sp = static_cast<float*>(softmax_y.data());
        for (int64_t i = 0; i < y.numel(); ++i) sp[i] = std::exp(yp[i]);

        Tensor row_sum = Tensor::empty(Shape{outer}, y.dtype(), y.device());
        const float* gp = static_cast<const float*>(gy.data());
        float* rsp = static_cast<float*>(row_sum.data());
        for (int64_t r = 0; r < outer; ++r) {
            float s = 0.0f;
            for (int64_t j = 0; j < last; ++j) s += gp[r * last + j];
            rsp[r] = s;
        }

        Tensor grad_x = Tensor::empty(y.shape(), y.dtype(), y.device());
        float* gxp = static_cast<float*>(grad_x.data());
        for (int64_t r = 0; r < outer; ++r) {
            float s = rsp[r];
            for (int64_t j = 0; j < last; ++j) {
                int64_t idx = r * last + j;
                gxp[idx] = gp[idx] - sp[idx] * s;
            }
        }
        return {grad_x};
    }

private:
    SavedTensor y_;
};

} // namespace tensorforge
