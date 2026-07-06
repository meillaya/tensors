#pragma once

#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"

#include <cstddef>
#include <vector>

namespace tensorforge {

[[nodiscard]] Tensor reshape(const Tensor& tensor, Shape new_shape);

[[nodiscard]] Tensor transpose(const Tensor& tensor, size_t dim0, size_t dim1);

[[nodiscard]] Tensor permute(const Tensor& tensor, std::vector<size_t> dims);

} // namespace tensorforge
