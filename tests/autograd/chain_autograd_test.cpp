#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"
#include "autograd/AutogradMeta.hpp"

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Tensor;
using tensorforge::full;

namespace {

void check_all_equal(const Tensor& t, float expected) {
    const float* ptr = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        CHECK(ptr[i] == doctest::Approx(expected).epsilon(0.001));
    }
}

}  // namespace

TEST_CASE("chain autograd propagates requires_grad through add-mul-sum") {
    Tensor a = full({2, 3}, 2.0f, Dtype::Float32, Device::cpu());
    Tensor b = full({2, 3}, 3.0f, Dtype::Float32, Device::cpu());
    a.requires_grad(true);
    b.requires_grad(true);

    Tensor c = a + b;
    REQUIRE(c.requires_grad() == true);
    REQUIRE(c.autograd_meta() != nullptr);
    REQUIRE(c.autograd_meta()->grad_fn_ != nullptr);

    Tensor d = c * a;
    REQUIRE(d.requires_grad() == true);
    REQUIRE(d.autograd_meta()->grad_fn_ != nullptr);

    Tensor e = d.sum(0);
    REQUIRE(e.requires_grad() == true);
    REQUIRE(e.autograd_meta()->grad_fn_ != nullptr);

    e.backward();

    REQUIRE(a.grad().numel() == 6);
    REQUIRE(b.grad().numel() == 6);

    // d = (a+b)*a, backward grad starts as all 1s (sum)
    // dd/da = (a+b) + a*1 = 2a + b = 2*2 + 3 = 7
    check_all_equal(a.grad(), 7.0f);
    // dd/db = a*1 = 2
    check_all_equal(b.grad(), 2.0f);
}

TEST_CASE("transpose propagates requires_grad and backward reaches leaf") {
    Tensor W = full({3, 4}, 0.5f, Dtype::Float32, Device::cpu());
    W.requires_grad(true);

    Tensor Wt = W.transpose(0, 1);
    REQUIRE(Wt.requires_grad() == true);
    REQUIRE(Wt.autograd_meta() != nullptr);
    REQUIRE(Wt.autograd_meta()->grad_fn_ != nullptr);

    Wt.backward();
    REQUIRE(W.grad().numel() == 12);
    check_all_equal(W.grad(), 1.0f);
}

TEST_CASE("matmul over transposed weight propagates grad to leaf") {
    Tensor W = full({3, 4}, 0.5f, Dtype::Float32, Device::cpu());
    W.requires_grad(true);

    Tensor Wt = W.transpose(0, 1);  // [4, 3]
    Tensor x = full({2, 4}, 1.0f, Dtype::Float32, Device::cpu());

    Tensor logits = x.matmul(Wt);  // [2, 3]
    REQUIRE(logits.requires_grad() == true);
    REQUIRE(logits.autograd_meta()->grad_fn_ != nullptr);

    logits.backward();
    REQUIRE(W.grad().numel() == 12);
    REQUIRE(W.grad().shape().ndim() == 2);
    REQUIRE(W.grad().shape()[0] == 3);
    REQUIRE(W.grad().shape()[1] == 4);
}

TEST_CASE("Linear-style forward+backward chain end-to-end") {
    Tensor W = full({3, 4}, 0.5f, Dtype::Float32, Device::cpu());
    W.requires_grad(true);

    Tensor Wt = W.transpose(0, 1);  // [4, 3]
    Tensor x = full({2, 4}, 1.0f, Dtype::Float32, Device::cpu());
    Tensor h = x.matmul(Wt);        // [2, 3]
    Tensor logits = h.relu();       // [2, 3]
    Tensor loss = logits.sum(0);    // [3]

    REQUIRE(loss.requires_grad() == true);
    REQUIRE(loss.autograd_meta()->grad_fn_ != nullptr);

    loss.backward();
    REQUIRE(W.grad().numel() == 12);
}
