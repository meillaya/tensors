// TensorForge - 2D matrix transpose CUDA kernel header (Wave 6 / T28)
//
// Computes out[i, j] = in[j, i] for row-major 2D matrices of shape
// [M, N] -> [N, M]. Used by nn::Conv2d::backward to materialize the
// transposed weight matrix for grad_input computation.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

namespace tensorforge {

void launch_transpose_2d(const void* in, void* out, int64_t M, int64_t N,
                          Dtype dtype, void* stream);

}  // namespace tensorforge
