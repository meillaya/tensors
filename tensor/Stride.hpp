#pragma once

#include "tensor/Shape.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace tensorforge {

class Stride {
public:
    Stride() = default;

    Stride(std::initializer_list<int64_t> strides) : strides_(strides) {}

    explicit Stride(std::vector<int64_t> strides) : strides_(std::move(strides)) {}

    [[nodiscard]] int64_t operator[](size_t i) const { return strides_[i]; }

    [[nodiscard]] int64_t& operator[](size_t i) { return strides_[i]; }

    [[nodiscard]] size_t ndim() const noexcept { return strides_.size(); }

    [[nodiscard]] const std::vector<int64_t>& data() const noexcept { return strides_; }

    [[nodiscard]] std::vector<int64_t>& data() noexcept { return strides_; }

    [[nodiscard]] bool operator==(const Stride& other) const noexcept {
        return strides_ == other.strides_;
    }

    [[nodiscard]] bool operator!=(const Stride& other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] static Stride compute_row_major(const Shape& shape) {
        const size_t n = shape.ndim();
        std::vector<int64_t> strides(n);
        if (n == 0) {
            return Stride{};
        }
        strides[n - 1] = 1;
        for (size_t i = n - 1; i-- > 0;) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return Stride(std::move(strides));
    }

private:
    std::vector<int64_t> strides_;
};

inline std::ostream& operator<<(std::ostream& os, const Stride& stride) {
    os << "Stride(";
    for (size_t i = 0; i < stride.ndim(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << stride[i];
    }
    os << ")";
    return os;
}

} // namespace tensorforge
