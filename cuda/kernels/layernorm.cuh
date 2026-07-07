// TensorForge — layer-norm kernel header (Wave 4 / T21)
//
// Row-wise layer normalization: for each row of a [rows, cols] input,
// compute mean and variance, then output
//
//     out[r, c] = (in[r, c] - mean[r]) / sqrt(var[r] + eps) * gamma[c] + beta[c]
//
// gamma and beta are 1D [cols] vectors and must be on the same device as
// the input. `eps` defaults to 1e-5 in the launcher's typed wrappers.
//
// Dtype is templated; all reductions are done in FP32 internally.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

namespace tensorforge {

// Layer-norm over the last dim of a 2D [rows, cols] row-major tensor.
// gamma and beta must be [cols] vectors. Caller must ensure rows*cols > 0.
void launch_layernorm(const void* in, const void* gamma, const void* beta,
                      void* out, int64_t rows, int64_t cols,
                      float eps, Dtype dtype, void* stream);

}  // namespace tensorforge
