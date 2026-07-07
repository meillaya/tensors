// TensorForge - col2im CUDA kernel header (Wave 6 / T28)
//
// Inverse of im2col: scatter-add entries from a column matrix of shape
// [N, C*kH*kW, outH*outW] back into an input tensor of shape [N, C, H, W].
// Used by nn::Conv2d::backward to fold grad_input from a column-matrix
// gradient into the original input layout.
//
// All operands live on the same device. `stream` may be nullptr for the
// default stream. `input_grad` is *added to* in place - the caller is
// responsible for zeroing it before calling launch_col2im if a fresh
// gradient buffer is desired.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

namespace tensorforge {

void launch_col2im(const void* col, void* input_grad,
                    int64_t N, int64_t C, int64_t H, int64_t W,
                    int64_t kH, int64_t kW,
                    int64_t stride_h, int64_t stride_w,
                    int64_t pad_h, int64_t pad_w,
                    int64_t dilation_h, int64_t dilation_w,
                    int64_t outH, int64_t outW,
                    Dtype dtype, void* stream);

}  // namespace tensorforge
