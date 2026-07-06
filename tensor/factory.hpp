#pragma once

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>

namespace tensorforge {

[[nodiscard]] Tensor zeros(Shape shape, Dtype dtype, Device device);

[[nodiscard]] Tensor ones(Shape shape, Dtype dtype, Device device);

[[nodiscard]] Tensor full(Shape shape, double value, Dtype dtype, Device device);

[[nodiscard]] Tensor arange(int64_t start, int64_t end, int64_t step, Dtype dtype, Device device);

[[nodiscard]] Tensor copy(const Tensor& src, Device dst_device);

} // namespace tensorforge
