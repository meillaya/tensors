#include "nn/CrossEntropyLoss.hpp"

#include "tensor/factory.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace tensorforge::nn {

Tensor CrossEntropyLoss::forward(Tensor logits, Tensor targets) {
    // logits: (N, C) Float32; targets: (N,) Int64.
    if (logits.dtype() != Dtype::Float32) {
        throw std::invalid_argument(
            "CrossEntropyLoss: logits must be Float32 in v1");
    }
    if (logits.shape().ndim() != 2) {
        throw std::invalid_argument(
            "CrossEntropyLoss: logits must be 2D (N, C) in v1");
    }
    if (targets.dtype() != Dtype::Int64) {
        throw std::invalid_argument(
            "CrossEntropyLoss: targets must be Int64 in v1");
    }

    const int64_t N = logits.shape()[0];
    const int64_t C = logits.shape()[1];
    if (targets.shape().numel() != N) {
        throw std::invalid_argument(
            "CrossEntropyLoss: targets.numel() must equal logits.shape()[0]");
    }

    // Softmax along last dim gives P(class | sample). We then take
    // -log(P[target]) and average — numerically equivalent to
    // log_softmax + NLL but reuses the existing softmax kernel.
    Tensor probs = logits.softmax(static_cast<int64_t>(-1)); // (N, C)

    // Per-sample NLL: gather probs[i, targets[i]], clamp to avoid log(0),
    // negate, then average.
    Tensor losses = Tensor::empty(Shape{N}, Dtype::Float32, Device::cpu());
    Tensor probs_cpu = probs.to(Device::cpu());
    Tensor targets_cpu = targets.to(Device::cpu());

    const float* probs_p = static_cast<const float*>(probs_cpu.data());
    const int64_t* t_p = static_cast<const int64_t*>(targets_cpu.data());
    float* loss_p = static_cast<float*>(losses.data());

    for (int64_t i = 0; i < N; ++i) {
        const int64_t t = t_p[i];
        if (t < 0 || t >= C) {
            throw std::out_of_range(
                "CrossEntropyLoss: target index out of range");
        }
        const float p = probs_p[i * C + t];
        // Clamp to keep log() finite even when softmax underflows.
        const float clamped = p < 1e-9f ? 1e-9f : p;
        loss_p[i] = -std::log(clamped);
    }

    // Move per-sample losses back to logits' device and reduce.
    losses = losses.to(logits.device());
    return losses.mean();
}

Tensor CrossEntropyLoss::forward(Tensor /*logits*/) {
    // Required by Module's pure-virtual signature, but
    // CrossEntropyLoss semantically needs targets. Refuse loudly
    // rather than silently producing a wrong answer.
    throw std::logic_error(
        "CrossEntropyLoss::forward(logits) requires targets; "
        "call forward(logits, targets) instead.");
}

} // namespace tensorforge::nn