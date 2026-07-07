#include "autograd/AccumulateGrad.hpp"

#include "autograd/AutogradMeta.hpp"

#include <cstdint>
#include <stdexcept>
#include <cstdio>

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
        meta->grad_ = grads[0];
    }
    return {grads[0]};
}

} // namespace tensorforge
