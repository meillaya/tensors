// TensorForge - reduce_sum_axis CUDA kernel header (Wave 6 / T28)
//
// Reduces a 4D NCHW tensor over dims 0, 2, 3 keeping dim 1, so the
// output has shape [C] with out[c] = sum over (n, h, w) of x[n, c, h, w].
// Used by nn::Conv2d::backward to compute grad_bias from grad_output.
//
// All operands live on the same device. `stream` may be nullptr.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

namespace tensorforge {

void launch_reduce_sum_axis_nchw(const void* x, void* out,
                                  int64_t N, int64_t C, int64_t H, int64_t W,
                                  Dtype dtype, void* stream);

}  // namespace tensorforge
