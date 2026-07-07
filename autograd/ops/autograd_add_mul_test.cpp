// TensorForge - autograd add/mul backward tests (Wave 7 / T33)
//
// Exercises Tensor::operator+ and Tensor::operator* end-to-end: wire the
// autograd graph via the wirer slot (g_wire_add / g_wire_mul), run the
// Engine, and check the gradients of each leaf. These are CPU-only paths
// since the wirer / Engine live in CPU code; tags [cpu][autograd][fp32].
// All test cases assert the actual analytic gradient of a scalar loss
// rather than treating the backward machinery as opaque. This way, if
// either the wirer or the underlying Node misbehaves, the failure
// points at the wrong gradient value, not just "graph didn't run".

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "autograd/AutogradMeta.hpp"
#include "autograd/Engine.hpp"
#include "autograd/ops/AddBackward.hpp"
#include "autograd/ops/MulBackward.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#include <cstdint>
#include <memory>
#include <vector>

using tensorforge::AddBackward;
using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Edge;
using tensorforge::Engine;
using tensorforge::MulBackward;
using tensorforge::Shape;
using tensorforge::Tensor;
using tensorforge::full;

namespace {

void check_all_equal(const Tensor& t, float expected, float tol = 1e-5f) {
    REQUIRE(t.dtype() == Dtype::Float32);
    REQUIRE(t.device() == Device::cpu());
    const float* ptr = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        CHECK(ptr[i] == doctest::Approx(expected).epsilon(tol));
    }
}

Tensor filled(Shape shape, float value) {
    Tensor t = Tensor::empty(shape, Dtype::Float32, Device::cpu());
    float* p = static_cast<float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) p[i] = value;
    return t;
}

Tensor arange_filled(int64_t n, float base, float step) {
    Tensor t = Tensor::empty(Shape{n}, Dtype::Float32, Device::cpu());
    float* p = static_cast<float*>(t.data());
    for (int64_t i = 0; i < n; ++i) p[i] = base + step * i;
    return t;
}

} // namespace

TEST_CASE("operator+ end-to-end: c = a + b -> dc/da = dc/db = 1") {
    Tensor a = filled({2, 3}, 2.0f);
    Tensor b = filled({2, 3}, 5.0f);
    a.requires_grad(true);
    b.requires_grad(true);

    Tensor c = a + b;
    REQUIRE(c.requires_grad());
    REQUIRE(c.autograd_meta()->grad_fn_ != nullptr);

    c.backward();
    check_all_equal(a.grad(), 1.0f);
    check_all_equal(b.grad(), 1.0f);
}

TEST_CASE("operator+ with non-grad inputs: result has no grad_fn") {
    Tensor a = filled({2, 3}, 1.0f);
    Tensor b = filled({2, 3}, 1.0f);
    Tensor c = a + b;
    CHECK_FALSE(c.requires_grad());
    CHECK(c.autograd_meta() == nullptr);
}

TEST_CASE("operator+ with one grad input still wires output") {
    Tensor a = filled({4}, 1.5f);
    Tensor b = filled({4}, 2.5f);
    a.requires_grad(true);

    Tensor c = a + b;
    CHECK(c.requires_grad());
    REQUIRE(c.autograd_meta()->grad_fn_ != nullptr);
    c.backward();
    check_all_equal(a.grad(), 1.0f);
    CHECK(b.grad().numel() == 0);
}

TEST_CASE("operator+ with arange inputs: gradient is still ones") {
    Tensor a = arange_filled(6, 0.0f, 1.0f);
    Tensor b = arange_filled(6, 6.0f, 1.0f);
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a + b;
    c.backward();
    check_all_equal(a.grad(), 1.0f);
    check_all_equal(b.grad(), 1.0f);
}

TEST_CASE("operator* end-to-end: c = a * b -> dc/da = b, dc/db = a") {
    Tensor a = filled({2, 3}, 2.0f);
    Tensor b = filled({2, 3}, 3.0f);
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a * b;
    REQUIRE(c.requires_grad());
    c.backward();
    check_all_equal(a.grad(), 3.0f);
    check_all_equal(b.grad(), 2.0f);
}

TEST_CASE("operator* with arange inputs: analytic gradient") {
    Tensor a = arange_filled(4, 0.0f, 1.0f);
    Tensor b = arange_filled(4, 4.0f, 1.0f);
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a * b;
    c.backward();
    {
        const float* ga = static_cast<const float*>(a.grad().data());
        CHECK(ga[0] == doctest::Approx(4.0f));
        CHECK(ga[1] == doctest::Approx(5.0f));
        CHECK(ga[2] == doctest::Approx(6.0f));
        CHECK(ga[3] == doctest::Approx(7.0f));
    }
    {
        const float* gb = static_cast<const float*>(b.grad().data());
        CHECK(gb[0] == doctest::Approx(0.0f));
        CHECK(gb[1] == doctest::Approx(1.0f));
        CHECK(gb[2] == doctest::Approx(2.0f));
        CHECK(gb[3] == doctest::Approx(3.0f));
    }
}

TEST_CASE("MulBackward errors when saved input was mutated in-place") {
    Tensor a = filled({2, 2}, 1.0f);
    Tensor b = filled({2, 2}, 2.0f);
    a.requires_grad(true);
    b.requires_grad(true);

    auto mul_node = std::make_shared<MulBackward>(a, b);
    mul_node->next_edges_ = {
        Edge(a.autograd_meta()->grad_accumulator_, 0),
        Edge(b.autograd_meta()->grad_accumulator_, 0),
    };

    (void)a.bump_version();

    Tensor grad = filled({2, 2}, 1.0f);
    Engine engine;
    CHECK_THROWS(engine.execute(mul_node, grad, false));
}

TEST_CASE("d = c * a with c = a + b: dd/da = 2a + b, dd/db = a") {
    Tensor a = filled({2, 3}, 2.0f);
    Tensor b = filled({2, 3}, 3.0f);
    a.requires_grad(true);
    b.requires_grad(true);

    Tensor c = a + b;
    Tensor d = c * a;
    d.backward();

    check_all_equal(a.grad(), 7.0f);
    check_all_equal(b.grad(), 2.0f);
}
