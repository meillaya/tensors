// TensorForge — softmax kernel header (Wave 4 / T20)
//
// Numerically stable 2-pass softmax: subtract max then exp/sum. Operates on
// a single 2D row layout `[rows, cols]` with one block per row. Dtype is
// templated; the accumulator is float for FP16/BF16 to avoid under/overflow
// while computing exp().
//
// launch_softmax is callable from g++ TUs (Tensor.cpp) via a void* stream
// pointer to avoid dragging cuda_runtime.h into the host header.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

namespace tensorforge {

// `in` and `out` point to row-major [rows, cols] blocks of dtype T. The
// caller must ensure rows*cols > 0.
void launch_softmax(const void* in, void* out, int64_t rows, int64_t cols,
                    Dtype dtype, void* stream);

}  // namespace tensorforge
