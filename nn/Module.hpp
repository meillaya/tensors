#pragma once

#include "tensor/Tensor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorforge::nn {

// Parameter wraps a Tensor with requires_grad=true.
//
// In v1, Parameter is a thin wrapper: the Tensor inside carries the
// requires_grad bit. We intentionally do NOT introduce a separate
// "Variable" type (PyTorch legacy) — Tensor is the variable.
class Parameter {
public:
    Tensor data_;

    Parameter() = default;
    explicit Parameter(Tensor t) : data_(std::move(t)) {
        data_.requires_grad(true);
    }

    Parameter(const Parameter&) = default;
    Parameter(Parameter&&) noexcept = default;
    Parameter& operator=(const Parameter&) = default;
    Parameter& operator=(Parameter&&) noexcept = default;
};

// Module — base class for all neural-network components.
//
// v1: holds a flat map of named parameters and a flat map of named
// submodules. parameters() recurses into submodules so a Sequential-
// style container (future) can compose layers without each layer
// needing to know about its parent.
class Module {
public:
    Module() = default;
    virtual ~Module() = default;

    // Register a parameter under a name. Existing entry with same
    // name is overwritten (matches PyTorch behavior).
    void register_parameter(const std::string& name, Parameter p);

    // Register a submodule under a name. Owning shared_ptr keeps
    // the submodule alive as long as the parent exists.
    void register_module(const std::string& name, std::shared_ptr<Module> m);

    // Forward — must be overridden by every concrete module.
    virtual Tensor forward(Tensor x) = 0;

    // operator() convenience.
    Tensor operator()(Tensor x) { return forward(std::move(x)); }

    // Collect all parameters in this module and all submodules (DFS).
    // Returned pointers are non-owning; valid as long as *this lives.
    std::vector<Parameter*> parameters();

    // Same as parameters() but with dotted names, e.g. "inner.w".
    // Useful for debugging and for the future optimizer wiring.
    std::vector<std::pair<std::string, Parameter*>> named_parameters(
        const std::string& prefix = "");

    // Training / eval flags. v1 only flips a bool; no dropout / BN
    // behavior is implemented yet.
    void train() { training_ = true; }
    void eval() { training_ = false; }
    [[nodiscard]] bool is_training() const noexcept { return training_; }

protected:
    std::unordered_map<std::string, Parameter> parameters_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;

private:
    bool training_ = true;
};

} // namespace tensorforge::nn