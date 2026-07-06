#pragma once

#include "autograd/Node.hpp"
#include "tensor/Tensor.hpp"

#include <memory>
#include <vector>

namespace tensorforge {

struct AutogradMeta;

// AccumulateGrad — leaf sink in the backward graph.
// Holds a weak_ptr to the leaf variable's AutogradMeta and writes the
// accumulated gradient into meta->grad_ when apply() is called.
// sequence_nr_ is set to UINT64_MAX so the engine schedules it last.
class AccumulateGrad : public Node {
public:
    explicit AccumulateGrad(std::weak_ptr<AutogradMeta> meta);

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;

private:
    std::weak_ptr<AutogradMeta> meta_;
};

} // namespace tensorforge
