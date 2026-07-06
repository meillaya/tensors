#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/Dtype.hpp"

using tensorforge::dtype_name;
using tensorforge::dtype_size;
using tensorforge::Dtype;

TEST_CASE("Dtype size lookup") {
    CHECK(dtype_size(Dtype::Float32) == 4);
    CHECK(dtype_size(Dtype::Float16) == 2);
    CHECK(dtype_size(Dtype::BFloat16) == 2);
    CHECK(dtype_size(Dtype::Int32) == 4);
    CHECK(dtype_size(Dtype::Int64) == 8);
    CHECK(dtype_size(Dtype::Bool) == 1);
}

TEST_CASE("Dtype name lookup") {
    CHECK(dtype_name(Dtype::Float32) == "Float32");
    CHECK(dtype_name(Dtype::Float16) == "Float16");
    CHECK(dtype_name(Dtype::BFloat16) == "BFloat16");
    CHECK(dtype_name(Dtype::Int32) == "Int32");
    CHECK(dtype_name(Dtype::Int64) == "Int64");
    CHECK(dtype_name(Dtype::Bool) == "Bool");
}

TEST_CASE("Dtype values are distinct") {
    CHECK(static_cast<uint8_t>(Dtype::Float32) != static_cast<uint8_t>(Dtype::Float16));
    CHECK(static_cast<uint8_t>(Dtype::Int64) != static_cast<uint8_t>(Dtype::Bool));
}
