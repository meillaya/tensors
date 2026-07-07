// TensorForge — im2col kernel tests (Wave 5 / T26)
//
// Verifies launch_im2col produces the correct column matrix layout:
//   - Trivial 1x1 kernel, 1x1 input, no padding (1→1 mapping)
//   - 3x3 kernel, 4x4 input, no padding: explicit hand-computed layout
//   - 3x3 kernel with same/zero padding (effectively 6x6 padded input
//     collapses to outH=outW=4)
//   - Stride 2 (output downsamples 2x)
//   - Dilation 2 on 3x3 kernel (samples every other input pixel)
//   - FP16 / BF16 dtypes within tolerance

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/im2col.cuh"
#include "tensor/Dtype.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using tensorforge::Device;
using tensorforge::DeviceContext;
using tensorforge::Dtype;
using tensorforge::Tensor;

namespace {

void sync_stream() {
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
}

Tensor upload_fp32(const float* host, int64_t n_elements, Dtype dtype) {
    Tensor t = Tensor::empty({n_elements}, dtype, Device::cuda(0));
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(DeviceContext::current().current_stream);
    switch (dtype) {
    case Dtype::Float32:
        cudaMemcpyAsync(t.data(), host, sizeof(float) * n_elements,
                        cudaMemcpyHostToDevice, stream);
        break;
    case Dtype::Float16: {
        std::vector<__half> casted(n_elements);
        for (int64_t i = 0; i < n_elements; ++i) casted[i] = __float2half(host[i]);
        cudaMemcpyAsync(t.data(), casted.data(), sizeof(__half) * n_elements,
                        cudaMemcpyHostToDevice, stream);
        break;
    }
    case Dtype::BFloat16: {
        std::vector<__nv_bfloat16> casted(n_elements);
        for (int64_t i = 0; i < n_elements; ++i) {
            casted[i] = __float2bfloat16_rn(host[i]);
        }
        cudaMemcpyAsync(t.data(), casted.data(), sizeof(__nv_bfloat16) * n_elements,
                        cudaMemcpyHostToDevice, stream);
        break;
    }
    default:
        throw std::invalid_argument("upload_fp32: unsupported dtype");
    }
    sync_stream();
    return t;
}

// Reference im2col on host. Layout matches the kernel: row-major, shape
// (N, C * kH * kW, outH * outW).
std::vector<float> ref_im2col(const std::vector<float>& in,
                               int64_t N, int64_t C, int64_t H, int64_t W,
                               int64_t kH, int64_t kW,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w,
                               int64_t dilation_h, int64_t dilation_w) {
    int64_t outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    int64_t total = N * C * kH * kW * outH * outW;
    std::vector<float> out(total, 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ki = 0; ki < kH; ++ki) {
                for (int64_t kj = 0; kj < kW; ++kj) {
                    for (int64_t oh = 0; oh < outH; ++oh) {
                        for (int64_t ow = 0; ow < outW; ++ow) {
                            int64_t ih = oh * stride_h + ki * dilation_h - pad_h;
                            int64_t iw = ow * stride_w + kj * dilation_w - pad_w;
                            float v = 0.0f;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                v = in[((n * C + c) * H + ih) * W + iw];
                            }
                            int64_t row = (c * kH + ki) * kW + kj;
                            int64_t out_idx = oh * outW + ow;
                            out[(n * (C * kH * kW) + row) * (outH * outW) + out_idx] = v;
                        }
                    }
                }
            }
        }
    }
    return out;
}

void run_im2col(const Tensor& in, Tensor& col,
                int64_t N, int64_t C, int64_t H, int64_t W,
                int64_t kH, int64_t kW,
                int64_t stride_h, int64_t stride_w,
                int64_t pad_h, int64_t pad_w,
                int64_t dilation_h, int64_t dilation_w,
                Dtype dtype) {
    auto& ctx = tensorforge::DeviceContext::current();
    sync_stream();
    tensorforge::launch_im2col(in.data(), col.data(),
                                N, C, H, W, kH, kW,
                                stride_h, stride_w, pad_h, pad_w,
                                dilation_h, dilation_w,
                                dtype, (void*)ctx.current_stream);
    sync_stream();
}

}  // namespace

TEST_CASE("[gpu][fp32] im2col 1x1 kernel, 1x1 input, no padding") {
    constexpr int64_t N = 1, C = 1, H = 1, W = 1;
    constexpr int64_t kH = 1, kW = 1;
    float in[1] = {42.0f};

    auto expected = ref_im2col({42.0f}, N, C, H, W, kH, kW,
                                1, 1, 0, 0, 1, 1);

    Tensor in_t = upload_fp32(in, 1, Dtype::Float32);
    Tensor col = Tensor::empty({N, C * kH * kW, 1 * 1},
                                Dtype::Float32, Device::cuda(0));
    run_im2col(in_t, col, N, C, H, W, kH, kW, 1, 1, 0, 0, 1, 1,
                Dtype::Float32);

    std::vector<float> out(1);
    cudaMemcpy(out.data(), col.data(), sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(out[0] == doctest::Approx(42.0f).epsilon(1e-6));
}

TEST_CASE("[gpu][fp32] im2col 3x3 kernel, 4x4 input, no padding") {
    constexpr int64_t N = 1, C = 1, H = 4, W = 4;
    constexpr int64_t kH = 3, kW = 3;
    // 4x4 input → 2x2 output with 3x3 kernel (no pad, stride 1).
    std::vector<float> in(H * W);
    for (int i = 0; i < H * W; ++i) in[i] = static_cast<float>(i);
    auto expected = ref_im2col(in, N, C, H, W, kH, kW, 1, 1, 0, 0, 1, 1);

    Tensor in_t = upload_fp32(in.data(), in.size(), Dtype::Float32);
    Tensor col = Tensor::empty({N, C * kH * kW, 2 * 2},
                                Dtype::Float32, Device::cuda(0));
    run_im2col(in_t, col, N, C, H, W, kH, kW, 1, 1, 0, 0, 1, 1,
                Dtype::Float32);

    std::vector<float> out(1 * 9 * 4);
    cudaMemcpy(out.data(), col.data(), sizeof(float) * out.size(),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("[gpu][fp32] im2col 3x3 kernel with zero padding (same conv)") {
    // same-padding: pad=1, k=3, stride=1 → in H=W=4 → out H=W=4
    constexpr int64_t N = 1, C = 2, H = 4, W = 4;
    constexpr int64_t kH = 3, kW = 3;
    std::vector<float> in(N * C * H * W);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i + 1);
    auto expected = ref_im2col(in, N, C, H, W, kH, kW, 1, 1, 1, 1, 1, 1);

    Tensor in_t = upload_fp32(in.data(), in.size(), Dtype::Float32);
    Tensor col = Tensor::empty({N, C * kH * kW, 4 * 4},
                                Dtype::Float32, Device::cuda(0));
    run_im2col(in_t, col, N, C, H, W, kH, kW, 1, 1, 1, 1, 1, 1,
                Dtype::Float32);

    std::vector<float> out(N * C * kH * kW * 4 * 4);
    cudaMemcpy(out.data(), col.data(), sizeof(float) * out.size(),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("[gpu][fp32] im2col 3x3 kernel stride=2 (downsamples 2x)") {
    constexpr int64_t N = 1, C = 1, H = 5, W = 5;
    constexpr int64_t kH = 3, kW = 3;
    // outH = (5 + 0 - 2 - 1)/2 + 1 = 2
    std::vector<float> in(H * W);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
    auto expected = ref_im2col(in, N, C, H, W, kH, kW, 2, 2, 0, 0, 1, 1);

    Tensor in_t = upload_fp32(in.data(), in.size(), Dtype::Float32);
    Tensor col = Tensor::empty({N, C * kH * kW, 2 * 2},
                                Dtype::Float32, Device::cuda(0));
    run_im2col(in_t, col, N, C, H, W, kH, kW, 2, 2, 0, 0, 1, 1,
                Dtype::Float32);

    std::vector<float> out(1 * 9 * 4);
    cudaMemcpy(out.data(), col.data(), sizeof(float) * out.size(),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("[gpu][fp32] im2col 3x3 kernel dilation=2") {
    constexpr int64_t N = 1, C = 1, H = 5, W = 5;
    constexpr int64_t kH = 3, kW = 3;
    // dilation=2: effective kernel span = 1 + 2*(3-1) = 5
    // outH = (5 + 0 - 4 - 1)/1 + 1 = 1
    std::vector<float> in(H * W);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);
    auto expected = ref_im2col(in, N, C, H, W, kH, kW, 1, 1, 0, 0, 2, 2);

    Tensor in_t = upload_fp32(in.data(), in.size(), Dtype::Float32);
    Tensor col = Tensor::empty({N, C * kH * kW, 1 * 1},
                                Dtype::Float32, Device::cuda(0));
    run_im2col(in_t, col, N, C, H, W, kH, kW, 1, 1, 0, 0, 2, 2,
                Dtype::Float32);

    std::vector<float> out(9);
    cudaMemcpy(out.data(), col.data(), sizeof(float) * 9,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < 9; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("[gpu][fp32] im2col batch N=2 verifies per-sample independence") {
    constexpr int64_t N = 2, C = 1, H = 3, W = 3;
    constexpr int64_t kH = 2, kW = 2;
    std::vector<float> in(N * H * W);
    for (int64_t i = 0; i < N * H * W; ++i) {
        in[i] = static_cast<float>(i + 1);
    }
    auto expected = ref_im2col(in, N, C, H, W, kH, kW, 1, 1, 0, 0, 1, 1);

    Tensor in_t = upload_fp32(in.data(), in.size(), Dtype::Float32);
    Tensor col = Tensor::empty({N, C * kH * kW, 2 * 2},
                                Dtype::Float32, Device::cuda(0));
    run_im2col(in_t, col, N, C, H, W, kH, kW, 1, 1, 0, 0, 1, 1,
                Dtype::Float32);

    int64_t total = N * C * kH * kW * 2 * 2;
    std::vector<float> out(total);
    cudaMemcpy(out.data(), col.data(), sizeof(float) * total,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < total; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-6));
    }
}

TEST_CASE("[gpu][fp16] im2col FP16 matches FP32 reference") {
    constexpr int64_t N = 1, C = 1, H = 4, W = 4;
    constexpr int64_t kH = 3, kW = 3;
    std::vector<float> in(H * W);
    for (int i = 0; i < H * W; ++i) in[i] = static_cast<float>(i);
    auto expected = ref_im2col(in, N, C, H, W, kH, kW, 1, 1, 1, 1, 1, 1);

    Tensor in_t = upload_fp32(in.data(), in.size(), Dtype::Float16);
    Tensor col = Tensor::empty({N, C * kH * kW, 4 * 4},
                                Dtype::Float16, Device::cuda(0));
    run_im2col(in_t, col, N, C, H, W, kH, kW, 1, 1, 1, 1, 1, 1,
                Dtype::Float16);

    std::vector<__half> out(N * C * kH * kW * 4 * 4);
    cudaMemcpy(out.data(), col.data(), sizeof(__half) * out.size(),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(__half2float(out[i]) == doctest::Approx(expected[i]).epsilon(0.05));
    }
}

TEST_CASE("[gpu][bf16] im2col BF16 matches FP32 reference") {
    constexpr int64_t N = 1, C = 1, H = 4, W = 4;
    constexpr int64_t kH = 3, kW = 3;
    std::vector<float> in(H * W);
    for (int i = 0; i < H * W; ++i) in[i] = static_cast<float>(i);
    auto expected = ref_im2col(in, N, C, H, W, kH, kW, 1, 1, 1, 1, 1, 1);

    Tensor in_t = upload_fp32(in.data(), in.size(), Dtype::BFloat16);
    Tensor col = Tensor::empty({N, C * kH * kW, 4 * 4},
                                Dtype::BFloat16, Device::cuda(0));
    run_im2col(in_t, col, N, C, H, W, kH, kW, 1, 1, 1, 1, 1, 1,
                Dtype::BFloat16);

    std::vector<__nv_bfloat16> out(N * C * kH * kW * 4 * 4);
    cudaMemcpy(out.data(), col.data(), sizeof(__nv_bfloat16) * out.size(),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(__bfloat162float(out[i]) == doctest::Approx(expected[i]).epsilon(0.1));
    }
}