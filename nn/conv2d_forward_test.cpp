// TensorForge — Conv2d forward tests (Wave 6 / T27)
//
// Verifies nn::Conv2d::forward on:
//   - 1x1 kernel identity conv (output = input per-channel, scaled by weight)
//   - 1-channel 3x3 kernel with known-input / known-weight golden values
//   - same-padding 3x3 kernel (outH = outW = inH = inW)
//   - stride=2 downsampling
//   - multi-channel (Cin=3, Cout=2)
//   - batch N=2 verifies per-sample independence
//   - bias-disabled mode
//
// All tests compare the kernel output against a host-side reference
// convolution that handles stride/pad/dilation explicitly.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/CudaContext.hpp"
#include "nn/Conv2d.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using tensorforge::Device;
using tensorforge::DeviceContext;
using tensorforge::Dtype;
using tensorforge::Tensor;
using tensorforge::full;
using tensorforge::nn::Conv2d;

namespace {

void sync_stream() {
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
}

// Reference convolution on host: same semantics as the im2col+GEMM
// pipeline. Used as a cross-check on the GPU forward output.
std::vector<float> ref_conv2d(const std::vector<float>& in,
                                const std::vector<float>& w,
                                const std::vector<float>* bias,
                                int64_t N, int64_t Cin, int64_t H, int64_t W,
                                int64_t Cout, int64_t kH, int64_t kW,
                                int64_t stride_h, int64_t stride_w,
                                int64_t pad_h, int64_t pad_w,
                                int64_t dilation_h, int64_t dilation_w) {
    int64_t outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    std::vector<float> out(N * Cout * outH * outW, 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t co = 0; co < Cout; ++co) {
            for (int64_t oh = 0; oh < outH; ++oh) {
                for (int64_t ow = 0; ow < outW; ++ow) {
                    float acc = bias ? (*bias)[co] : 0.0f;
                    for (int64_t ci = 0; ci < Cin; ++ci) {
                        for (int64_t ki = 0; ki < kH; ++ki) {
                            for (int64_t kj = 0; kj < kW; ++kj) {
                                int64_t ih = oh * stride_h + ki * dilation_h - pad_h;
                                int64_t iw = ow * stride_w + kj * dilation_w - pad_w;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float v = in[((n * Cin + ci) * H + ih) * W + iw];
                                    float kv = w[((co * Cin + ci) * kH + ki) * kW + kj];
                                    acc += v * kv;
                                }
                            }
                        }
                    }
                    out[((n * Cout + co) * outH + oh) * outW + ow] = acc;
                }
            }
        }
    }
    return out;
}

// Helper: upload a known FP32 weight tensor directly (so we can pin
// weight values per-test instead of relying on Conv2d's init).
void upload_weight(Conv2d& conv, const std::vector<float>& host, int64_t Cout,
                    int64_t Cin, int64_t kH, int64_t kW) {
    Tensor& w = conv.weight().data_;
    cudaMemcpyAsync(w.data(), host.data(), sizeof(float) * host.size(),
                     cudaMemcpyHostToDevice,
                     reinterpret_cast<cudaStream_t>(DeviceContext::current().current_stream));
    sync_stream();
}

void upload_bias(Conv2d& conv, const std::vector<float>& host) {
    Tensor& b = conv.bias().data_;
    cudaMemcpyAsync(b.data(), host.data(), sizeof(float) * host.size(),
                     cudaMemcpyHostToDevice,
                     reinterpret_cast<cudaStream_t>(DeviceContext::current().current_stream));
    sync_stream();
}

void upload_input(Tensor& t, const std::vector<float>& host) {
    cudaMemcpyAsync(t.data(), host.data(), sizeof(float) * host.size(),
                     cudaMemcpyHostToDevice,
                     reinterpret_cast<cudaStream_t>(DeviceContext::current().current_stream));
    sync_stream();
}

}  // namespace

TEST_CASE("[gpu][fp32] conv2d_forward 1x1 kernel identity") {
    // Cin=Cout=1, 1x1 kernel with weight=2.0 -> out = 2 * in.
    Conv2d conv(/*Cin*/1, /*Cout*/1, /*kH*/1, /*kW*/1,
                 /*stride*/1, 1, /*pad*/0, 0, /*dil*/1, 1,
                 /*bias*/false);
    upload_weight(conv, {2.0f}, 1, 1, 1, 1);

    std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f};  // 1x1x2x2
    Tensor x = Tensor::empty({1, 1, 2, 2}, Dtype::Float32, Device::cuda(0));
    upload_input(x, in);

    Tensor y = conv.forward(x);
    sync_stream();

    CHECK(y.shape()[0] == 1);
    CHECK(y.shape()[1] == 1);
    CHECK(y.shape()[2] == 2);
    CHECK(y.shape()[3] == 2);

    std::vector<float> out(4);
    cudaMemcpy(out.data(), y.data(), sizeof(float) * 4, cudaMemcpyDeviceToHost);
    std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f};
    for (int i = 0; i < 4; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-4));
    }
}

TEST_CASE("[gpu][fp32] conv2d_forward 3x3 same-padding golden case") {
    // Cin=1, Cout=1, 3x3 kernel, same padding (pad=1). We can compute the
    // expected output by hand for a tiny input.
    constexpr int64_t N = 1, Cin = 1, H = 3, W = 3;
    constexpr int64_t Cout = 1, kH = 3, kW = 3;
    constexpr int64_t pad = 1;

    Conv2d conv(Cin, Cout, kH, kW, 1, 1, pad, pad, 1, 1, /*bias*/true);

    // Weight: all 1.0; bias: 0.0 (Conv2d inits weight=0.1, bias=0.05).
    // We'll override weight and bias for this test.
    std::vector<float> w(kH * kW, 1.0f);
    upload_weight(conv, w, Cout, Cin, kH, kW);
    std::vector<float> b = {0.0f};
    upload_bias(conv, b);

    std::vector<float> in = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload_input(x, in);

    auto expected = ref_conv2d(in, w, &b, N, Cin, H, W, Cout, kH, kW,
                                1, 1, pad, pad, 1, 1);

    Tensor y = conv.forward(x);
    sync_stream();
    int64_t outH = 3, outW = 3;
    std::vector<float> out(N * Cout * outH * outW);
    cudaMemcpy(out.data(), y.data(), sizeof(float) * out.size(),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-3));
    }
}

TEST_CASE("[gpu][fp32] conv2d_forward stride=2 downsamples 2x") {
    // Cin=1, Cout=1, 3x3 kernel, stride=2, no padding.
    // Input 4x4 -> output 1x1 (since (4 - 3)/2 + 1 = 1).
    constexpr int64_t N = 1, Cin = 1, H = 4, W = 4;
    constexpr int64_t Cout = 1, kH = 3, kW = 3;

    Conv2d conv(Cin, Cout, kH, kW, /*stride*/2, 2, /*pad*/0, 0, 1, 1, false);
    std::vector<float> w = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    upload_weight(conv, w, Cout, Cin, kH, kW);

    std::vector<float> in(H * W);
    for (int i = 0; i < H * W; ++i) in[i] = static_cast<float>(i + 1);

    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload_input(x, in);

    auto expected = ref_conv2d(in, w, nullptr, N, Cin, H, W, Cout, kH, kW,
                                2, 2, 0, 0, 1, 1);

    Tensor y = conv.forward(x);
    sync_stream();
    CHECK(y.shape()[2] == 1);
    CHECK(y.shape()[3] == 1);
    int64_t total = N * Cout * 1 * 1;
    std::vector<float> out(total);
    cudaMemcpy(out.data(), y.data(), sizeof(float) * total,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < total; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-3));
    }
}

TEST_CASE("[gpu][fp32] conv2d_forward multi-channel Cin=3 Cout=2") {
    constexpr int64_t N = 1, Cin = 3, H = 4, W = 4;
    constexpr int64_t Cout = 2, kH = 3, kW = 3;

    Conv2d conv(Cin, Cout, kH, kW, 1, 1, 1, 1, 1, 1, /*bias*/true);

    // 9 weights per (cout, cin) pair, for a total of Cout*Cin*kH*kW.
    std::vector<float> w(Cout * Cin * kH * kW);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = 0.01f * static_cast<float>((i % 13) + 1);
    }
    upload_weight(conv, w, Cout, Cin, kH, kW);
    std::vector<float> b = {0.5f, -0.25f};
    upload_bias(conv, b);

    std::vector<float> in(N * Cin * H * W);
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = 0.1f * static_cast<float>((i % 7) + 1);
    }
    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload_input(x, in);

    auto expected = ref_conv2d(in, w, &b, N, Cin, H, W, Cout, kH, kW,
                                1, 1, 1, 1, 1, 1);

    Tensor y = conv.forward(x);
    sync_stream();
    int64_t outH = 4, outW = 4;
    int64_t total = N * Cout * outH * outW;
    std::vector<float> out(total);
    cudaMemcpy(out.data(), y.data(), sizeof(float) * total,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < total; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-3));
    }
}

TEST_CASE("[gpu][fp32] conv2d_forward batch N=2 per-sample independence") {
    constexpr int64_t N = 2, Cin = 1, H = 3, W = 3;
    constexpr int64_t Cout = 1, kH = 2, kW = 2;

    Conv2d conv(Cin, Cout, kH, kW, 1, 1, 0, 0, 1, 1, /*bias*/false);
    std::vector<float> w = {1.0f, 2.0f, 3.0f, 4.0f};  // Cout*Cin*kH*kW
    upload_weight(conv, w, Cout, Cin, kH, kW);

    std::vector<float> in(N * H * W);
    for (int64_t i = 0; i < N * H * W; ++i) {
        in[i] = static_cast<float>(i + 1);
    }
    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload_input(x, in);

    auto expected = ref_conv2d(in, w, nullptr, N, Cin, H, W, Cout, kH, kW,
                                1, 1, 0, 0, 1, 1);

    Tensor y = conv.forward(x);
    sync_stream();
    int64_t outH = 2, outW = 2;
    int64_t total = N * Cout * outH * outW;
    std::vector<float> out(total);
    cudaMemcpy(out.data(), y.data(), sizeof(float) * total,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < total; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-4));
    }
}

TEST_CASE("[gpu][fp32] conv2d_forward no-bias path") {
    constexpr int64_t N = 1, Cin = 1, H = 3, W = 3;
    constexpr int64_t Cout = 1, kH = 2, kW = 2;

    Conv2d conv(Cin, Cout, kH, kW, 1, 1, 0, 0, 1, 1, /*bias*/false);
    std::vector<float> w = {0.5f, 0.5f, 0.5f, 0.5f};
    upload_weight(conv, w, Cout, Cin, kH, kW);

    std::vector<float> in = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload_input(x, in);

    auto expected = ref_conv2d(in, w, nullptr, N, Cin, H, W, Cout, kH, kW,
                                1, 1, 0, 0, 1, 1);

    Tensor y = conv.forward(x);
    sync_stream();
    int64_t outH = 2, outW = 2;
    int64_t total = N * Cout * outH * outW;
    std::vector<float> out(total);
    cudaMemcpy(out.data(), y.data(), sizeof(float) * total,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < total; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-4));
    }
}