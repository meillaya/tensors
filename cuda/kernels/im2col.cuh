// TensorForge — im2col kernel header (Wave 5 / T26)
//
// Converts a [N, C, H, W] 4D input into a [N, C*kH*kW, outH*outW] 3D
// column matrix suitable for GEMM-based convolution. Each column
// corresponds to one (output_spatial_position) and the rows enumerate
// (input_channel, kernel_height, kernel_width) — i.e. each column is the
// kH*kW*C "patch" of the input that the corresponding output element sees.
//
// Supports padding (pad_h, pad_w), stride (stride_h, stride_w), and
// dilation (dilation_h, dilation_w). Out-of-bounds input positions
// (those that lie in the padded halo) are written as zero — the GEMM
// downstream never sees NaN/garbage from padding.
//
// Output layout: row-major, shape (N, C * kH * kW, outH * outW).
//
// `stream` may be nullptr for the default stream.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

namespace tensorforge {

// im2col forward.
//
// Input shape:    [N, C, H, W]
// Output shape:   [N, C * kH * kW, outH * outW]
//   where:
//     outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1
//     outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1
//
// Caller must pre-allocate `col` with the correct dtype and at least
// N * C * kH * kW * outH * outW elements. All buffers on same device.
void launch_im2col(const void* input, void* col,
                   int64_t N, int64_t C, int64_t H, int64_t W,
                   int64_t kH, int64_t kW,
                   int64_t stride_h, int64_t stride_w,
                   int64_t pad_h, int64_t pad_w,
                   int64_t dilation_h, int64_t dilation_w,
                   Dtype dtype, void* stream);

}  // namespace tensorforge