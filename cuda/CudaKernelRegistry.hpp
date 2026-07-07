// TensorForge — host-only CUDA kernel dispatch registration (Wave 4 / T17+)
//
// Function pointers registered by //cuda/kernels at startup so that
// tensor/Tensor.cpp (compiled by g++) can call CUDA kernels without
// including cuda_runtime.h.

#pragma once

#include "tensor/Dtype.hpp"

#include <cstdint>

namespace tensorforge {

using BinaryOpFn = void (*)(const void*, const void*, void*, int64_t, Dtype, void*);
using UnaryOpFn = void (*)(const void*, void*, int64_t, Dtype, void*);
using LeakyReluFn = void (*)(const void*, void*, int64_t, Dtype, float, void*);

void register_cuda_add(BinaryOpFn fn);
void register_cuda_mul(BinaryOpFn fn);
void register_cuda_relu(UnaryOpFn fn);
void register_cuda_sigmoid(UnaryOpFn fn);
void register_cuda_tanh(UnaryOpFn fn);
void register_cuda_leaky_relu(LeakyReluFn fn);

void register_current_stream(void* stream);

} // namespace tensorforge