#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "nn/Activations.hpp"
#include "tensor/Tensor.hpp"

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Tensor;

using tensorforge::nn::ReLU;
using tensorforge::nn::Sigmoid;
using tensorforge::nn::Tanh;

// v1 stubs only assert that forward() runs and returns a tensor with
// the expected numel. Numerical correctness and backward wiring are
// out of scope for T39 (T19 elementwise + T33-T35 autograd-bwd).
TEST_CASE("ReLU module exists and forward() returns Tensor") {
    ReLU relu;
    Tensor x = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());

    Tensor y = relu.forward(x);
    CHECK(y.numel() == 6);
}

TEST_CASE("Sigmoid module exists") {
    Sigmoid s;
    Tensor x = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());

    Tensor y = s.forward(x);
    CHECK(y.numel() == 6);
}

TEST_CASE("Tanh module exists") {
    Tanh t;
    Tensor x = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());

    Tensor y = t.forward(x);
    CHECK(y.numel() == 6);
}

TEST_CASE("operator() forwards through forward()") {
    ReLU relu;
    Tensor x = Tensor::empty(Shape{4}, Dtype::Float32, Device::cpu());

    Tensor y = relu(std::move(x));
    CHECK(y.numel() == 4);
}