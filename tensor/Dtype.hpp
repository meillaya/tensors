#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tensorforge {

enum class Dtype : uint8_t {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    Bool,
};

[[nodiscard]] inline constexpr size_t dtype_size(Dtype dtype) noexcept {
    switch (dtype) {
    case Dtype::Float32:
        return 4;
    case Dtype::Float16:
        return 2;
    case Dtype::BFloat16:
        return 2;
    case Dtype::Int32:
        return 4;
    case Dtype::Int64:
        return 8;
    case Dtype::Bool:
        return 1;
    }
    return 0;
}

[[nodiscard]] inline constexpr std::string_view dtype_name(Dtype dtype) noexcept {
    switch (dtype) {
    case Dtype::Float32:
        return "Float32";
    case Dtype::Float16:
        return "Float16";
    case Dtype::BFloat16:
        return "BFloat16";
    case Dtype::Int32:
        return "Int32";
    case Dtype::Int64:
        return "Int64";
    case Dtype::Bool:
        return "Bool";
    }
    return "Unknown";
}

} // namespace tensorforge
