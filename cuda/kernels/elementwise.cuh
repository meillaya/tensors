// TensorForge — elementwise kernels: add, mul, activations (Wave 4 / T17-T19)

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

// This header is includable from both .cc (g++) and .cu (nvcc) TUs. It does
// NOT include cuda_runtime.h directly. The stream is passed as void* so
// there is no type-name collision between the two toolchains' cudaStream_t
// typedefs.

namespace tensorforge {

void launch_add(const void* a, const void* b, void* out, int64_t n,
                Dtype dtype, void* stream);

void launch_mul(const void* a, const void* b, void* out, int64_t n,
                Dtype dtype, void* stream);

void launch_relu(const void* in, void* out, int64_t n, Dtype dtype, void* stream);
void launch_leaky_relu(const void* in, void* out, int64_t n, Dtype dtype,
                        float alpha, void* stream);
void launch_sigmoid(const void* in, void* out, int64_t n, Dtype dtype, void* stream);
void launch_tanh(const void* in, void* out, int64_t n, Dtype dtype, void* stream);

} // namespace tensorforge