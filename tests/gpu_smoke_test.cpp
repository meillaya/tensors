#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("GPU availability check" * doctest::skip(true)) {
    // Real implementation in Wave 3 (T14: CUDA stream wrapper)
    // For now, this test is skipped
    MESSAGE("GPU smoke placeholder — real test in T14");
}
