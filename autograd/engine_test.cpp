#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "autograd/ops/AddBackward.hpp"
#include "autograd/AutogradMeta.hpp"
#include "autograd/Engine.hpp"
#include "autograd/Node.hpp"
#include "tensor/Tensor.hpp"

#include <memory>
#include <vector>

using tensorforge::AddBackward;
using tensorforge::AutogradMeta;
using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Edge;
using tensorforge::Engine;
using tensorforge::Node;
using tensorforge::Shape;
using tensorforge::Tensor;

namespace {

Tensor make_filled(Shape shape, float value) {
    Tensor t = Tensor::empty(shape, Dtype::Float32, Device::cpu());
    float* ptr = static_cast<float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        ptr[i] = value;
    }
    return t;
}

void check_all_equal(const Tensor& t, float expected) {
    const float* ptr = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        CHECK(ptr[i] == expected);
    }
}

} // namespace

TEST_CASE("Simple chain: c = a + b; backward produces a.grad == b.grad == ones") {
    Tensor a = make_filled({2, 3}, 1.0f);
    a.requires_grad(true);

    Tensor b = make_filled({2, 3}, 1.0f);
    b.requires_grad(true);

    auto add_node = std::make_shared<AddBackward>();
    add_node->next_edges_ = {
        Edge(a.autograd_meta()->grad_accumulator_, 0),
        Edge(b.autograd_meta()->grad_accumulator_, 0),
    };

    Tensor grad = make_filled({2, 3}, 1.0f);
    Engine engine;
    engine.execute(add_node, grad, false);

    CHECK(a.grad().numel() == 6);
    check_all_equal(a.grad(), 1.0f);
    check_all_equal(b.grad(), 1.0f);
}

TEST_CASE("Branched: e=a+b, f=a+c, loss=e+f; a.grad == 2, b.grad == 1, c.grad == 1") {
    Tensor a = make_filled({2, 3}, 1.0f);
    a.requires_grad(true);

    Tensor b = make_filled({2, 3}, 1.0f);
    b.requires_grad(true);

    Tensor c = make_filled({2, 3}, 1.0f);
    c.requires_grad(true);

    auto add_e = std::make_shared<AddBackward>();
    add_e->next_edges_ = {
        Edge(a.autograd_meta()->grad_accumulator_, 0),
        Edge(b.autograd_meta()->grad_accumulator_, 0),
    };

    auto add_f = std::make_shared<AddBackward>();
    add_f->next_edges_ = {
        Edge(a.autograd_meta()->grad_accumulator_, 0),
        Edge(c.autograd_meta()->grad_accumulator_, 0),
    };

    auto add_loss = std::make_shared<AddBackward>();
    add_loss->next_edges_ = {
        Edge(add_e, 0),
        Edge(add_f, 0),
    };

    Tensor grad = make_filled({2, 3}, 1.0f);
    Engine engine;
    engine.execute(add_loss, grad, false);

    check_all_equal(a.grad(), 2.0f);
    check_all_equal(b.grad(), 1.0f);
    check_all_equal(c.grad(), 1.0f);
}

TEST_CASE("Tensor::backward() runs engine on grad_fn") {
    Tensor a = make_filled({2, 3}, 1.0f);
    a.requires_grad(true);

    Tensor b = make_filled({2, 3}, 1.0f);
    b.requires_grad(true);

    auto add_node = std::make_shared<AddBackward>();
    add_node->next_edges_ = {
        Edge(a.autograd_meta()->grad_accumulator_, 0),
        Edge(b.autograd_meta()->grad_accumulator_, 0),
    };

    Tensor output = make_filled({2, 3}, 2.0f);
    output.autograd_meta() = std::make_shared<AutogradMeta>();
    output.autograd_meta()->grad_fn_ = add_node;
    output.autograd_meta()->requires_grad_ = true;

    output.backward();

    check_all_equal(a.grad(), 1.0f);
    check_all_equal(b.grad(), 1.0f);
}

TEST_CASE("Deep chain: w = x + y; z = w + x; backward accumulates correctly") {
    Tensor x = make_filled({1}, 1.0f);
    x.requires_grad(true);

    Tensor y = make_filled({1}, 1.0f);
    y.requires_grad(true);

    auto add_w = std::make_shared<AddBackward>();
    add_w->next_edges_ = {
        Edge(x.autograd_meta()->grad_accumulator_, 0),
        Edge(y.autograd_meta()->grad_accumulator_, 0),
    };

    auto add_z = std::make_shared<AddBackward>();
    add_z->next_edges_ = {
        Edge(add_w, 0),
        Edge(x.autograd_meta()->grad_accumulator_, 0),
    };

    Tensor grad = make_filled({1}, 1.0f);
    Engine engine;
    engine.execute(add_z, grad, false);

    // z = (x + y) + x = 2x + y  →  dz/dx = 2, dz/dy = 1
    check_all_equal(x.grad(), 2.0f);
    check_all_equal(y.grad(), 1.0f);
}
