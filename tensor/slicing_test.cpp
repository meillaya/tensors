#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/slicing.hpp"

#include "tensor/Tensor.hpp"

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Stride;
using tensorforge::Tensor;

TEST_CASE("slice on each dimension") {
    Tensor t = Tensor::empty({4, 5, 6}, Dtype::Float32, Device::cpu());

    Tensor s0 = tensorforge::slice(t, 0, 1, 3);
    CHECK(s0.shape() == Shape{2, 5, 6});
    CHECK(s0.stride() == Stride{30, 6, 1});
    CHECK(s0.storage_offset() == 30);

    Tensor s1 = tensorforge::slice(t, 1, 0, 2);
    CHECK(s1.shape() == Shape{4, 2, 6});
    CHECK(s1.stride() == Stride{30, 6, 1});
    CHECK(s1.storage_offset() == 0);

    Tensor s2 = tensorforge::slice(t, 2, 2, 5);
    CHECK(s2.shape() == Shape{4, 5, 3});
    CHECK(s2.stride() == Stride{30, 6, 1});
    CHECK(s2.storage_offset() == 2);
}

TEST_CASE("slice negative indices") {
    Tensor t = Tensor::empty({4, 5, 6}, Dtype::Float32, Device::cpu());
    Tensor s = tensorforge::slice(t, 0, -3, -1);
    CHECK(s.shape() == Shape{2, 5, 6});
    CHECK(s.storage_offset() == 30);
}

TEST_CASE("slice shares storage") {
    Tensor t = Tensor::empty({4, 5}, Dtype::Float32, Device::cpu());
    const long use_count_before = t.storage().use_count();
    Tensor s = tensorforge::slice(t, 0, 1, 3);
    const long use_count_after = t.storage().use_count();
    CHECK(use_count_after == use_count_before + 1);
    CHECK(t.storage() == s.storage());
}

TEST_CASE("select reduces dimension") {
    Tensor t = Tensor::empty({4, 5, 6}, Dtype::Float32, Device::cpu());
    Tensor sel = tensorforge::select(t, 1, 2);
    CHECK(sel.shape() == Shape{4, 6});
    CHECK(sel.stride() == Stride{30, 1});
    CHECK(sel.storage_offset() == 12);

    Tensor sel_neg = tensorforge::select(t, 1, -1);
    CHECK(sel_neg.shape() == Shape{4, 6});
    CHECK(sel_neg.storage_offset() == 24);
}

TEST_CASE("narrow is slice by length") {
    Tensor t = Tensor::empty({4, 5, 6}, Dtype::Float32, Device::cpu());
    Tensor n = tensorforge::narrow(t, 0, 1, 2);
    CHECK(n.shape() == Shape{2, 5, 6});
    CHECK(n.storage_offset() == 30);
}

TEST_CASE("slice out of range throws") {
    Tensor t = Tensor::empty({4, 5}, Dtype::Float32, Device::cpu());
    CHECK_THROWS(tensorforge::slice(t, 0, 0, 5));
    CHECK_THROWS(tensorforge::slice(t, 0, 3, 2));
    CHECK_THROWS(tensorforge::slice(t, 2, 0, 1));
    CHECK_THROWS(tensorforge::select(t, 0, 4));
    CHECK_THROWS(tensorforge::select(t, 0, -5));
}
