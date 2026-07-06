#pragma once

#include "tensor/Tensor.hpp"

#include <mdspan/mdspan.hpp>

#include <cstddef>
#include <cstdint>

namespace tensorforge {

[[nodiscard]] Tensor slice(const Tensor& tensor, size_t dim, int64_t start, int64_t end);

[[nodiscard]] Tensor select(const Tensor& tensor, size_t dim, int64_t index);

[[nodiscard]] Tensor narrow(const Tensor& tensor, size_t dim, int64_t start, int64_t length);

} // namespace tensorforge
