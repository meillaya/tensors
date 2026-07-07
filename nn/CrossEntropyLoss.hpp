#pragma once

#include "nn/Module.hpp"
#include "tensor/Tensor.hpp"

namespace tensorforge::nn {

// CrossEntropyLoss — combines log_softmax + negative log-likelihood.
//
// Forward signature:
//   forward(logits, targets)
//     logits:  (N, C) Float32  — raw class scores
//     targets: (N,)   Int64    — ground-truth class indices
//   Returns a scalar Tensor (shape (1,)) holding the mean loss.
//
// v1 is forward-only and CPU Float32. The 1-arg `forward(Tensor)`
// override (required by Module) throws — call sites must supply targets.
class CrossEntropyLoss : public Module {
public:
    CrossEntropyLoss() = default;

    // Real entry point: logits + targets.
    [[nodiscard]] Tensor forward(Tensor logits, Tensor targets);

    // Module base override. CrossEntropyLoss cannot be invoked without
    // targets, so the 1-arg form is a programming-error guard.
    [[nodiscard]] Tensor forward(Tensor logits) override;
};

} // namespace tensorforge::nn