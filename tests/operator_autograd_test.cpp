#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"
#include "autograd/AutogradMeta.hpp"
#include "autograd/Engine.hpp"

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Tensor;
using tensorforge::full;

namespace {

void check_all_equal(const Tensor& t, float expected) {
    const float* ptr = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        CHECK(ptr[i] == expected);
    }
}

} // namespace

TEST_CASE("operator+ wires autograd backward") {
    Tensor a = full({2, 3}, 2.0, Dtype::Float32, Device::cpu());
    Tensor b = full({2, 3}, 3.0, Dtype::Float32, Device::cpu());
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a + b;
    REQUIRE(c.requires_grad() == true);
    REQUIRE(c.autograd_meta() != nullptr);
    REQUIRE(c.autograd_meta()->grad_fn_ != nullptr);
    c.backward();
    check_all_equal(a.grad(), 1.0f);
    check_all_equal(b.grad(), 1.0f);
}

TEST_CASE("operator* wires autograd backward") {
    Tensor a = full({2, 3}, 2.0, Dtype::Float32, Device::cpu());
    Tensor b = full({2, 3}, 3.0, Dtype::Float32, Device::cpu());
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a * b;
    REQUIRE(c.requires_grad() == true);
    c.backward();
    check_all_equal(a.grad(), 3.0f);
    check_all_equal(b.grad(), 2.0f);
}

TEST_CASE("chained operators propagate autograd") {
    Tensor a = full({2, 3}, 2.0, Dtype::Float32, Device::cpu());
    Tensor b = full({2, 3}, 3.0, Dtype::Float32, Device::cpu());
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a + b;       // c.grad_fn = AddBackward
    Tensor d = c * a;       // d.grad_fn = MulBackward, next_edges include c
    d.backward();
    // d = (a+b)*a, backward grad starts as all 1s
    // dd/da = c + a * dc/da = (a+b) + a * 1 = 2a + b
    // For a=2, b=3: dd/da = 4 + 3 = 7
    check_all_equal(a.grad(), 7.0f);
    // dd/db = a * dc/db = 2 * 1 = 2
    check_all_equal(b.grad(), 2.0f);
}