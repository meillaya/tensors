#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/Shape.hpp"
#include "tensor/Stride.hpp"

using tensorforge::Shape;
using tensorforge::Stride;

TEST_CASE("Stride construction and accessors") {
    Stride st{12, 4, 1};
    CHECK(st.ndim() == 3);
    CHECK(st[0] == 12);
    CHECK(st[1] == 4);
    CHECK(st[2] == 1);
}

TEST_CASE("Stride row-major computation") {
    Shape s{2, 3, 4};
    Stride st = Stride::compute_row_major(s);
    CHECK(st.ndim() == 3);
    CHECK(st[0] == 12);
    CHECK(st[1] == 4);
    CHECK(st[2] == 1);
    CHECK(st == Stride{12, 4, 1});
}

TEST_CASE("Stride row-major for 1D and empty shapes") {
    Shape s1{5};
    Stride st1 = Stride::compute_row_major(s1);
    CHECK(st1.ndim() == 1);
    CHECK(st1[0] == 1);

    Shape empty{};
    Stride st0 = Stride::compute_row_major(empty);
    CHECK(st0.ndim() == 0);
}
