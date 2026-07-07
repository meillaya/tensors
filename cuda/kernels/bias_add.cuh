// TensorForge — bias_add CUDA kernel header (Wave 6 / T27)
//
// In-place per-channel bias add: out[n, c, m, ...] += bias[c] for every
// spatial position and every batch sample. Dtype-templated, FP32
// accumulation. Stream-aware.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

namespace tensorforge {

// Adds a per-channel bias vector to a tensor whose second axis is the
// channel axis (NCHW / NCDHW layout). All operands must live on the
// same device. `stream` may be nullptr for the default stream.
void launch_bias_add(void* x, const void* bias,
                      int64_t N, int64_t C, int64_t M,
                      Dtype dtype, void* stream);

}  // namespace tensorforge
