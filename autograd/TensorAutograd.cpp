#include "tensor/Tensor.hpp"

#include "autograd/AccumulateGrad.hpp"
#include "autograd/AutogradMeta.hpp"
#include "autograd/Engine.hpp"

#include <stdexcept>

namespace tensorforge {

std::shared_ptr<AutogradMeta>& Tensor::autograd_meta() noexcept {
    return autograd_meta_;
}

const std::shared_ptr<AutogradMeta>& Tensor::autograd_meta() const noexcept {
    return autograd_meta_;
}

void Tensor::requires_grad(bool req) {
    if (!autograd_meta_) {
        autograd_meta_ = std::make_shared<AutogradMeta>();
    }
    autograd_meta_->requires_grad_ = req;
    if (req) {
        auto acc = std::make_shared<AccumulateGrad>(autograd_meta_);
        autograd_meta_->grad_accumulator_ = acc;
    }
}

bool Tensor::requires_grad() const noexcept {
    return autograd_meta_ && autograd_meta_->requires_grad_;
}

Tensor Tensor::grad() const {
    if (autograd_meta_) {
        return autograd_meta_->grad_;
    }
    return Tensor{};
}

void Tensor::backward() {
    if (!autograd_meta_ || !autograd_meta_->grad_fn_) {
        throw std::runtime_error("backward() called on tensor without grad_fn");
    }
    Tensor grad = Tensor::empty(shape(), dtype(), device());
    if (dtype() == Dtype::Float32) {
        float* ptr = static_cast<float*>(grad.data());
        for (int64_t i = 0; i < grad.numel(); ++i) {
            ptr[i] = 1.0f;
        }
    }
    Engine engine;
    engine.execute(autograd_meta_->grad_fn_, grad, false);
}

} // namespace tensorforge
