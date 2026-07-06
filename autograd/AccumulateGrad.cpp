#include "autograd/AccumulateGrad.hpp"

#include "autograd/AutogradMeta.hpp"

#include <cstdint>
#include <stdexcept>

namespace tensorforge {

AccumulateGrad::AccumulateGrad(std::weak_ptr<AutogradMeta> meta)
    : meta_(std::move(meta)) {
    sequence_nr_ = UINT64_MAX;
}

std::vector<Tensor> AccumulateGrad::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("AccumulateGrad::apply received no gradients");
    }
    if (auto meta = meta_.lock()) {
        meta->grad_ = std::move(grads[0]);
    }
    return {};
}

} // namespace tensorforge
