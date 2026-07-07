#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "nn/Linear.hpp"
#include "tensor/factory.hpp"

using namespace tensorforge;

TEST_CASE("Linear output shape") {
    nn::Linear fc(3, 2);
    Tensor x = ones({4, 3}, Dtype::Float32, Device::cpu());
    Tensor y = fc.forward(x);
    CHECK(y.shape()[0] == 4);
    CHECK(y.shape()[1] == 2);
}

TEST_CASE("Linear without bias") {
    nn::Linear fc(3, 2, /*bias=*/false);
    Tensor x = ones({4, 3}, Dtype::Float32, Device::cpu());
    Tensor y = fc.forward(x);
    CHECK(y.shape()[0] == 4);
    CHECK(y.shape()[1] == 2);
}

TEST_CASE("Linear gradient check") {
    nn::Linear fc(3, 2);
    Tensor x = full({2, 3}, 1.0f, Dtype::Float32, Device::cpu());
    x.requires_grad(true);
    Tensor y = fc.forward(x);
    y.sum(0).sum(0).backward();
    // x.grad should be sum of weight rows (broadcast)
    CHECK(x.grad().numel() == 6);
}
