#include "tensor/Tensor.hpp"

#include "autograd/AccumulateGrad.hpp"
#include "autograd/AutogradMeta.hpp"

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
    // Implemented in T32 (Engine).
}

} // namespace tensorforge
