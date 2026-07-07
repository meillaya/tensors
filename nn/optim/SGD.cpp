#include "nn/optim/SGD.hpp"

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace tensorforge::nn::optim {

SGD::SGD(std::vector<Tensor*> parameters, float lr, float momentum)
    : parameters_(std::move(parameters)),
      lr_(lr),
      momentum_(momentum) {
    if (lr_ <= 0.0f) {
        throw std::invalid_argument("SGD: lr must be positive");
    }
    if (momentum_ < 0.0f || momentum_ >= 1.0f) {
        throw std::invalid_argument("SGD: momentum must lie in [0, 1)");
    }

    velocity_.resize(parameters_.size());
    if (momentum_ > 0.0f) {
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            const Tensor& p = *parameters_[i];
            velocity_[i] = zeros(p.shape(), p.dtype(), p.device());
        }
    }
}

void SGD::step() {
    // Scalar `lr` expressed as a 1-element tensor so we can stay on the
    // operator* (Tensor x Tensor) path; v1 has no scalar overload.
    const Tensor lr_tensor = full(Shape{1}, static_cast<double>(lr_),
                                  Dtype::Float32, Device::cpu());

    for (std::size_t i = 0; i < parameters_.size(); ++i) {
        Tensor& p = *parameters_[i];
        const Tensor g_raw = p.grad();
        if (g_raw.numel() == 0) {
            continue;  // no gradient accumulated yet — leave p untouched
        }
        const Tensor g = g_raw.to(p.device());
        if (momentum_ > 0.0f) {
            // v = momentum * v + g
            const Tensor momentum_t = full(
                Shape{1}, static_cast<double>(momentum_),
                Dtype::Float32, Device::cpu());
            velocity_[i] = velocity_[i] * momentum_t + g;
            p = p - velocity_[i] * lr_tensor;
        } else {
            p = p - g * lr_tensor;
        }
    }
}

void SGD::zero_grad() {
    for (Tensor* p : parameters_) {
        if (!p) continue;
        const Tensor g = p->grad();
        if (g.numel() == 0) continue;
        if (g.dtype() != Dtype::Float32) continue;
        if (g.device().type != DeviceType::CPU) continue;
        // grad data() is const on a const Tensor; memset wants non-const.
        void* gp = const_cast<void*>(g.data());
        std::memset(gp, 0,
                    static_cast<std::size_t>(g.numel()) * sizeof(float));
    }
}

}  // namespace tensorforge::nn::optim
