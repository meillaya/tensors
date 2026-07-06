#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/factory.hpp"

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Tensor;

TEST_CASE("zeros creates tensor filled with zeros") {
    Tensor t = tensorforge::zeros({2, 3}, Dtype::Float32, Device::cpu());
    CHECK(t.shape() == Shape{2, 3});
    CHECK(t.numel() == 6);
    const float* data = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        CHECK(data[i] == 0.0f);
    }
}

TEST_CASE("ones creates tensor filled with ones") {
    Tensor t = tensorforge::ones({2, 3}, Dtype::Float32, Device::cpu());
    CHECK(t.shape() == Shape{2, 3});
    const float* data = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        CHECK(data[i] == 1.0f);
    }
}

TEST_CASE("full creates tensor filled with scalar value") {
    Tensor t = tensorforge::full({2, 3}, 7.0f, Dtype::Float32, Device::cpu());
    CHECK(t.shape() == Shape{2, 3});
    const float* data = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        CHECK(data[i] == 7.0f);
    }
}

TEST_CASE("arange creates sequential tensor") {
    Tensor t = tensorforge::arange(0, 5, 1, Dtype::Int64, Device::cpu());
    CHECK(t.shape() == Shape{5});
    const int64_t* data = static_cast<const int64_t*>(t.data());
    CHECK(data[0] == 0);
    CHECK(data[1] == 1);
    CHECK(data[2] == 2);
    CHECK(data[3] == 3);
    CHECK(data[4] == 4);
}

TEST_CASE("arange with non-unit step") {
    Tensor t = tensorforge::arange(0, 10, 3, Dtype::Int64, Device::cpu());
    CHECK(t.shape() == Shape{4});
    const int64_t* data = static_cast<const int64_t*>(t.data());
    CHECK(data[0] == 0);
    CHECK(data[1] == 3);
    CHECK(data[2] == 6);
    CHECK(data[3] == 9);
}

TEST_CASE("copy creates independent CPU tensor") {
    Tensor src = tensorforge::full({2, 3}, 3.0f, Dtype::Float32, Device::cpu());
    Tensor dst = tensorforge::copy(src, Device::cpu());
    CHECK(dst.shape() == src.shape());
    CHECK(dst.dtype() == src.dtype());
    CHECK(dst.device() == Device::cpu());
    CHECK(dst.storage() != src.storage());
    const float* data = static_cast<const float*>(dst.data());
    for (int64_t i = 0; i < dst.numel(); ++i) {
        CHECK(data[i] == 3.0f);
    }
}
