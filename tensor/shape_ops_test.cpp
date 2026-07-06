#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/factory.hpp"
#include "tensor/shape_ops.hpp"

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Stride;
using tensorforge::Tensor;

TEST_CASE("reshape view shares storage when contiguous") {
    Tensor t = tensorforge::zeros({2, 3, 4}, Dtype::Float32, Device::cpu());
    Tensor r = tensorforge::reshape(t, Shape{6, 4});
    CHECK(r.shape() == Shape{6, 4});
    CHECK(r.stride() == Stride{4, 1});
    CHECK(r.storage() == t.storage());
    CHECK(r.storage_offset() == t.storage_offset());
}

TEST_CASE("reshape rejects incompatible element count") {
    Tensor t = tensorforge::zeros({2, 3}, Dtype::Float32, Device::cpu());
    CHECK_THROWS(tensorforge::reshape(t, Shape{7}));
}

TEST_CASE("transpose swaps shape and stride") {
    Tensor t = tensorforge::zeros({2, 3, 4}, Dtype::Float32, Device::cpu());
    Tensor tr = tensorforge::transpose(t, 0, 2);
    CHECK(tr.shape() == Shape{4, 3, 2});
    CHECK(tr.stride() == Stride{1, 4, 12});
    CHECK(tr.storage() == t.storage());
}

TEST_CASE("transpose throws on invalid dimensions") {
    Tensor t = tensorforge::zeros({2, 3}, Dtype::Float32, Device::cpu());
    CHECK_THROWS(tensorforge::transpose(t, 0, 2));
}

TEST_CASE("permute reorders dimensions") {
    Tensor t = tensorforge::zeros({2, 3, 4}, Dtype::Float32, Device::cpu());
    Tensor p = tensorforge::permute(t, {2, 0, 1});
    CHECK(p.shape() == Shape{4, 2, 3});
    CHECK(p.stride() == Stride{1, 12, 4});
    CHECK(p.storage() == t.storage());
}

TEST_CASE("permute throws on invalid permutation") {
    Tensor t = tensorforge::zeros({2, 3, 4}, Dtype::Float32, Device::cpu());
    CHECK_THROWS(tensorforge::permute(t, {2, 0}));
    CHECK_THROWS(tensorforge::permute(t, {2, 0, 0}));
    CHECK_THROWS(tensorforge::permute(t, {3, 0, 1}));
}
