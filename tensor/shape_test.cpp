#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/Shape.hpp"

using tensorforge::Shape;

TEST_CASE("Shape construction and accessors") {
    Shape s{2, 3, 4};
    CHECK(s.ndim() == 3);
    CHECK(s[0] == 2);
    CHECK(s[1] == 3);
    CHECK(s[2] == 4);
    CHECK(s.numel() == 24);
}

TEST_CASE("Shape default and scalar") {
    Shape empty{};
    CHECK(empty.ndim() == 0);
    CHECK(empty.numel() == 0);

    Shape scalar{1};
    CHECK(scalar.ndim() == 1);
    CHECK(scalar.numel() == 1);
}

TEST_CASE("Shape equality and mutation") {
    Shape a{2, 3};
    Shape b{2, 3};
    Shape c{3, 2};
    CHECK(a == b);
    CHECK(a != c);

    a[1] = 5;
    CHECK(a[1] == 5);
    CHECK(a.numel() == 10);
}
