#pragma once

#include "autograd/IntrusivePtr.hpp"
#include "tensor/Tensor.hpp"

namespace tensorforge {

class Node;

// AutogradMeta — per-Tensor autograd state.
//
// v1 deviation: grad_accumulator_ is NodePtr (shared_ptr) rather than
// WeakNodePtr. AccumulateGrad holds weak_ptr<AutogradMeta> to break the
// would-be cycle (AutogradMeta → AccumulateGrad → AutogradMeta).
struct AutogradMeta {
    NodePtr<Node> grad_fn_;
    NodePtr<Node> grad_accumulator_;
    Tensor grad_;
    bool requires_grad_ = false;
};

} // namespace tensorforge
