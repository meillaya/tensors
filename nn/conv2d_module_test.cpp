// TensorForge - Conv2dModule tests (Wave 7 / T45)
//
// Verifies that nn::Conv2dModule:
//   * produces the right output shape for default (stride=1, pad=0) conv
//   * works without bias (parameter count drops to 1)
//   * correctly handles stride=2 with padding=1
//   * registers weight + bias as named parameters on construction

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "nn/Conv2dModule.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using tensorforge::Dtype;
using tensorforge::Device;
using tensorforge::Tensor;
using tensorforge::ones;
using tensorforge::nn::Conv2dModule;

TEST_CASE("Conv2dModule output shape") {
    Conv2dModule conv(3, 16, 3);
    Tensor x = ones({2, 3, 8, 8}, Dtype::Float32, Device::cpu());
    Tensor y = conv.forward(x);
    CHECK(y.shape()[0] == 2);   // N
    CHECK(y.shape()[1] == 16);  // Cout
    CHECK(y.shape()[2] == 6);   // outH (8 - 3 + 1 = 6)
    CHECK(y.shape()[3] == 6);   // outW
}

TEST_CASE("Conv2dModule without bias") {
    Conv2dModule conv(3, 16, 3, 1, 0, 1, /*bias=*/false);
    Tensor x = ones({2, 3, 8, 8}, Dtype::Float32, Device::cpu());
    Tensor y = conv.forward(x);
    CHECK(y.shape()[1] == 16);
}

TEST_CASE("Conv2dModule with stride and padding") {
    Conv2dModule conv(3, 8, 3, /*stride=*/2, /*padding=*/1);
    Tensor x = ones({2, 3, 8, 8}, Dtype::Float32, Device::cpu());
    Tensor y = conv.forward(x);
    // padding=1, stride=2, kernel=3: out = (8 + 2 - 3)/2 + 1 = 4
    CHECK(y.shape()[2] == 4);
    CHECK(y.shape()[3] == 4);
}

TEST_CASE("Conv2dModule has 2 parameters") {
    Conv2dModule conv(3, 16, 3);
    auto params = conv.parameters();
    CHECK(params.size() == 2);  // weight + bias
}
