#include "tensor/Tensor.hpp"

#include "autograd/AccumulateGrad.hpp"
#include "autograd/AutogradMeta.hpp"
#include "autograd/Engine.hpp"
#include "autograd/ops/AddBackward.hpp"
#include "autograd/ops/MulBackward.hpp"
#include "tensor/AutogradWirer.hpp"

#include <memory>
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

namespace {

void wire_add(Tensor& out, const Tensor& a, const Tensor& b) {
    if (!(a.requires_grad() || b.requires_grad())) {
        return;
    }
    out.requires_grad(true);
    auto grad_fn = std::make_shared<AddBackward>();
    out.autograd_meta()->grad_fn_ = grad_fn;
    grad_fn->next_edges_ = {
        Edge(a.autograd_meta() ? a.autograd_meta()->grad_accumulator_ : nullptr, 0),
        Edge(b.autograd_meta() ? b.autograd_meta()->grad_accumulator_ : nullptr, 0)
    };
    // Chain the AccumulateGrad to grad_fn so the engine walks past
    // intermediate tensors. Without this, d = (a+b)*a leaves a.grad
    // and b.grad unaccumulated for the (a+b) sub-graph.
    if (out.autograd_meta()->grad_accumulator_) {
        out.autograd_meta()->grad_accumulator_->next_edges_ = {
            Edge(grad_fn, 0)
        };
    }
}

void wire_mul(Tensor& out, const Tensor& a, const Tensor& b) {
    if (!(a.requires_grad() || b.requires_grad())) {
        return;
    }
    out.requires_grad(true);
    auto grad_fn = std::make_shared<MulBackward>(a, b);
    out.autograd_meta()->grad_fn_ = grad_fn;
    grad_fn->next_edges_ = {
        Edge(a.autograd_meta() ? a.autograd_meta()->grad_accumulator_ : nullptr, 0),
        Edge(b.autograd_meta() ? b.autograd_meta()->grad_accumulator_ : nullptr, 0)
    };
    if (out.autograd_meta()->grad_accumulator_) {
        out.autograd_meta()->grad_accumulator_->next_edges_ = {
            Edge(grad_fn, 0)
        };
    }
}

struct WirerRegistrar {
    WirerRegistrar() {
        register_add_wirer(&wire_add);
        register_mul_wirer(&wire_mul);
    }
};

WirerRegistrar g_wirer_registrar;

} // anonymous namespace

} // namespace tensorforge