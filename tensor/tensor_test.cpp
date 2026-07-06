#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/Tensor.hpp"

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Stride;
using tensorforge::Tensor;

TEST_CASE("Tensor::empty creates tensor with correct metadata") {
    Tensor t = Tensor::empty({2, 3}, Dtype::Float32, Device::cpu());
    CHECK(t.numel() == 6);
    CHECK(t.shape() == Shape{2, 3});
    CHECK(t.stride() == Stride{3, 1});
    CHECK(t.dtype() == Dtype::Float32);
    CHECK(t.device() == Device::cpu());
    CHECK(t.storage_offset() == 0);
}

TEST_CASE("Tensor::empty allocates aligned non-null data") {
    Tensor t = Tensor::empty({4, 4}, Dtype::Float32, Device::cpu());
    CHECK(t.data() != nullptr);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(t.data());
    CHECK(addr % 64 == 0);
}

TEST_CASE("Tensor version counter") {
    Tensor t = Tensor::empty({2, 3}, Dtype::Int64, Device::cpu());
    CHECK(t.version() == 0);
    CHECK(t.bump_version() == 1);
    CHECK(t.version() == 1);
    CHECK(t.bump_version() == 2);
}

TEST_CASE("Tensor default construction") {
    Tensor t;
    CHECK(t.numel() == 0);
    CHECK(t.data() == nullptr);
    CHECK(t.version() == 0);
}
