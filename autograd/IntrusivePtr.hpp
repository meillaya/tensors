#pragma once

#include <memory>

namespace tensorforge {

// NodePtr / WeakNodePtr — smart-pointer aliases for autograd graph nodes.
//
// v1 uses std::shared_ptr / std::weak_ptr. The autograd graph is a single
// DAG (no cycles), so shared_ptr ownership semantics are sufficient and
// avoid the complexity of a custom intrusive reference count. If profiling
// later shows that the atomic refcount overhead matters, we can switch to
// a custom IntrusivePtr without changing call sites (the alias is the
// single point of substitution).
template <typename T>
using NodePtr = std::shared_ptr<T>;

template <typename T>
using WeakNodePtr = std::weak_ptr<T>;

} // namespace tensorforge
