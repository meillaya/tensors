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
        if (grads[0].dtype() == Dtype::Float32 && grads[0].numel() > 0) {
            const float* p = static_cast<const float*>(grads[0].data());
            fprintf(stderr, "[AccumGrad] incoming: ");
            for (int64_t i = 0; i < grads[0].numel() && i < 8; ++i) {
                fprintf(stderr, "%.4f ", p[i]);
            }
            fprintf(stderr, "\n");
        }
        meta->grad_ = grads[0];
        if (meta->grad_.dtype() == Dtype::Float32 && meta->grad_.numel() > 0) {
            const float* p = static_cast<const float*>(meta->grad_.data());
            fprintf(stderr, "[AccumGrad] stored:   ");
            for (int64_t i = 0; i < meta->grad_.numel() && i < 8; ++i) {
                fprintf(stderr, "%.4f ", p[i]);
            }
            fprintf(stderr, "\n");
        }
    } else {
        fprintf(stderr, "[AccumGrad] lock FAILED\n");
    }
    return {grads[0]};
}

} // namespace tensorforge
