#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

namespace tensorforge {

class Shape {
public:
    Shape() = default;

    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}

    explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

    [[nodiscard]] int64_t operator[](size_t i) const { return dims_[i]; }

    [[nodiscard]] int64_t& operator[](size_t i) { return dims_[i]; }

    [[nodiscard]] size_t ndim() const noexcept { return dims_.size(); }

    [[nodiscard]] int64_t numel() const {
        if (dims_.empty()) {
            return 0;
        }
        return std::accumulate(dims_.begin(), dims_.end(), int64_t{1}, std::multiplies<int64_t>());
    }

    [[nodiscard]] const std::vector<int64_t>& data() const noexcept { return dims_; }

    [[nodiscard]] std::vector<int64_t>& data() noexcept { return dims_; }

    [[nodiscard]] bool operator==(const Shape& other) const noexcept {
        return dims_ == other.dims_;
    }

    [[nodiscard]] bool operator!=(const Shape& other) const noexcept {
        return !(*this == other);
    }

private:
    std::vector<int64_t> dims_;
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "Shape(";
    for (size_t i = 0; i < shape.ndim(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i];
    }
    os << ")";
    return os;
}

} // namespace tensorforge
