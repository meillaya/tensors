#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("smoke 1+1") {
    MESSAGE("running smoke 1+1");
    CHECK(1 + 1 == 2);
    CHECK(true);
}

TEST_CASE("smoke doctest tags work") {
    // This test should be filtered by tag
    CHECK(true);
}
