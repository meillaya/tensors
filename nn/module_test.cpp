#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "nn/Module.hpp"
#include "tensor/Tensor.hpp"

#include <memory>
#include <string>
#include <vector>

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Tensor;

using tensorforge::nn::Module;
using tensorforge::nn::Parameter;

namespace {

// Minimal linear-shaped layer for parameter iteration tests.
// Not a real Linear (T37) — just two parameters and a pass-through
// forward so we can construct an instance.
class TestLinearShim : public Module {
public:
    TestLinearShim() {
        register_parameter(
            "w",
            Parameter(Tensor::empty(Shape{2, 2}, Dtype::Float32, Device::cpu())));
        register_parameter(
            "b",
            Parameter(Tensor::empty(Shape{2}, Dtype::Float32, Device::cpu())));
    }

    Tensor forward(Tensor x) override { return x; }
};

// Single-parameter inner for the nested-module test.
class InnerShim : public Module {
public:
    InnerShim() {
        register_parameter(
            "w",
            Parameter(Tensor::empty(Shape{2, 2}, Dtype::Float32, Device::cpu())));
    }

    Tensor forward(Tensor x) override { return x; }
};

// Outer module: own parameter + one submodule.
class OuterShim : public Module {
public:
    OuterShim() {
        register_parameter(
            "b",
            Parameter(Tensor::empty(Shape{2}, Dtype::Float32, Device::cpu())));
        register_module("inner", std::make_shared<InnerShim>());
    }

    Tensor forward(Tensor x) override { return x; }
};

} // namespace

TEST_CASE("Parameter wraps Tensor with requires_grad") {
    Tensor t = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());
    REQUIRE(!t.requires_grad());

    Parameter p(std::move(t));
    CHECK(p.data_.requires_grad() == true);
}

TEST_CASE("Module registers and iterates parameters") {
    TestLinearShim m;

    auto params = m.parameters();
    CHECK(params.size() == 2);

    // Both must point to parameters that have requires_grad=true.
    bool all_require_grad = true;
    for (Parameter* p : params) {
        if (p == nullptr || !p->data_.requires_grad()) {
            all_require_grad = false;
            break;
        }
    }
    CHECK(all_require_grad);
}

TEST_CASE("Nested modules collect all parameters recursively") {
    OuterShim o;

    // Outer.b + Inner.w => 2 parameters total.
    auto params = o.parameters();
    CHECK(params.size() == 2);

    auto named = o.named_parameters();
    CHECK(named.size() == 2);

    // Collect names; expect exactly "b" and "inner.w".
    std::vector<std::string> names;
    names.reserve(named.size());
    for (const auto& kv : named) {
        names.push_back(kv.first);
    }

    bool has_b = false;
    bool has_inner_w = false;
    for (const auto& n : names) {
        if (n == "b") has_b = true;
        if (n == "inner.w") has_inner_w = true;
    }
    CHECK(has_b);
    CHECK(has_inner_w);
}

TEST_CASE("Module train/eval flag flips") {
    TestLinearShim m;
    CHECK(m.is_training() == true);

    m.eval();
    CHECK(m.is_training() == false);

    m.train();
    CHECK(m.is_training() == true);
}