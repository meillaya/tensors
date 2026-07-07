// TensorForge - Conv2d backward tests (Wave 6 / T28)
//
// Verifies nn::Conv2d::backward on:
//   - 1x1 kernel: analytic gradient matches host reference
//   - 2x2 stride-1: gradient vs host reference
//   - multi-channel Cin=2 Cout=3: full gradient check
//   - bias-disabled path
//   - shape validation: grad_input, grad_weight, grad_bias match expected

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
using tensorforge::nn::Conv2dGrad;

namespace {

void sync_stream() {
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
}

// Upload helpers for known FP32 values.
void upload(Tensor& t, const std::vector<float>& host) {
    cudaMemcpyAsync(t.data(), host.data(), sizeof(float) * host.size(),
                     cudaMemcpyHostToDevice,
                     reinterpret_cast<cudaStream_t>(DeviceContext::current().current_stream));
    sync_stream();
}

// Reference backward on host: given input, weight, bias, grad_output,
// compute grad_input / grad_weight / grad_bias by the textbook formula.
void ref_conv2d_backward(const std::vector<float>& in,
                          const std::vector<float>& w,
                          const float* bias,
                          const std::vector<float>& grad_out,
                          std::vector<float>& grad_in,
                          std::vector<float>& grad_w,
                          std::vector<float>& grad_b,
                          int64_t N, int64_t Cin, int64_t H, int64_t W,
                          int64_t Cout, int64_t kH, int64_t kW,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w,
                          int64_t dilation_h, int64_t dilation_w) {
    int64_t outH = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t outW = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    grad_in.assign(N * Cin * H * W, 0.0f);
    grad_w.assign(Cout * Cin * kH * kW, 0.0f);
    grad_b.assign(Cout, 0.0f);

    // grad_input: scatter-add. For each (n, co, oh, ow, ci, ki, kj):
    //   if (ih, iw) in-bounds, grad_in[n, ci, ih, iw] += grad_out[n, co, oh, ow] * w[co, ci, ki, kj]
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t co = 0; co < Cout; ++co) {
            for (int64_t oh = 0; oh < outH; ++oh) {
                for (int64_t ow = 0; ow < outW; ++ow) {
                    float go = grad_out[((n * Cout + co) * outH + oh) * outW + ow];
                    for (int64_t ci = 0; ci < Cin; ++ci) {
                        for (int64_t ki = 0; ki < kH; ++ki) {
                            for (int64_t kj = 0; kj < kW; ++kj) {
                                int64_t ih = oh * stride_h + ki * dilation_h - pad_h;
                                int64_t iw = ow * stride_w + kj * dilation_w - pad_w;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                float kv = w[((co * Cin + ci) * kH + ki) * kW + kj];
                                grad_in[((n * Cin + ci) * H + ih) * W + iw] += go * kv;
                                grad_w[((co * Cin + ci) * kH + ki) * kW + kj] +=
                                    go * in[((n * Cin + ci) * H + ih) * W + iw];
                            }
                        }
                    }
                }
            }
        }
    }

    // grad_bias
    if (bias) {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t co = 0; co < Cout; ++co) {
                for (int64_t oh = 0; oh < outH; ++oh) {
                    for (int64_t ow = 0; ow < outW; ++ow) {
                        grad_b[co] += grad_out[((n * Cout + co) * outH + oh) * outW + ow];
                    }
                }
            }
        }
    }
}

}  // namespace

TEST_CASE("[gpu][fp32] conv2d_backward 1x1 kernel gradient") {
    constexpr int64_t N = 1, Cin = 1, H = 2, W = 2;
    constexpr int64_t Cout = 1, kH = 1, kW = 1;
    Conv2d conv(Cin, Cout, kH, kW, 1, 1, 0, 0, 1, 1, /*bias*/false);

    // 1x1 kernel with weight=2.0. Output[i,j] = 2 * input[i,j].
    upload(conv.weight().data_, {2.0f});

    std::vector<float> in = {1, 2, 3, 4};
    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload(x, in);

    std::vector<float> grad_out = {0.1f, 0.2f, 0.3f, 0.4f};
    Tensor go = Tensor::empty({N, Cout, H, W}, Dtype::Float32, Device::cuda(0));
    upload(go, grad_out);

    auto grads = conv.backward(go, x);
    sync_stream();

    // grad_input = grad_out * w (no overlap with 1x1 kernel).
    CHECK(grads.grad_input.shape()[0] == N);
    CHECK(grads.grad_input.shape()[1] == Cin);
    CHECK(grads.grad_input.shape()[2] == H);
    CHECK(grads.grad_input.shape()[3] == W);
    CHECK(grads.grad_weight.shape()[0] == Cout);
    CHECK(grads.grad_weight.shape()[1] == Cin);
    CHECK(grads.grad_weight.shape()[2] == kH);
    CHECK(grads.grad_weight.shape()[3] == kW);
    CHECK(grads.grad_bias.shape()[0] == Cout);

    std::vector<float> gi(N * Cin * H * W);
    cudaMemcpy(gi.data(), grads.grad_input.data(),
               sizeof(float) * gi.size(), cudaMemcpyDeviceToHost);
    std::vector<float> expected_gi = {0.2f, 0.4f, 0.6f, 0.8f};
    for (size_t i = 0; i < gi.size(); ++i) {
        CHECK(gi[i] == doctest::Approx(expected_gi[i]).epsilon(1e-4));
    }

    // grad_weight[0,0,0,0] = sum over (n, oh, ow) of grad_out[n,0,oh,ow] * input[n,0,oh,ow]
    // = 0.1*1 + 0.2*2 + 0.3*3 + 0.4*4 = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
    std::vector<float> gw(Cout * Cin * kH * kW);
    cudaMemcpy(gw.data(), grads.grad_weight.data(),
               sizeof(float) * gw.size(), cudaMemcpyDeviceToHost);
    CHECK(gw[0] == doctest::Approx(3.0f).epsilon(1e-4));

    // grad_bias is zero (bias_enabled=false but the tensor is still allocated).
    std::vector<float> gb(Cout);
    cudaMemcpy(gb.data(), grads.grad_bias.data(),
               sizeof(float) * Cout, cudaMemcpyDeviceToHost);
    CHECK(gb[0] == doctest::Approx(0.0f).epsilon(1e-6));
}

TEST_CASE("[gpu][fp32] conv2d_backward 2x2 stride=1 multi-sample") {
    constexpr int64_t N = 2, Cin = 1, H = 3, W = 3;
    constexpr int64_t Cout = 2, kH = 2, kW = 2;
    Conv2d conv(Cin, Cout, kH, kW, 1, 1, 0, 0, 1, 1, /*bias*/true);

    std::vector<float> w = {1, 2, 3, 4,   // Cout=0
                             5, 6, 7, 8};  // Cout=1
    upload(conv.weight().data_, w);
    std::vector<float> b = {0.1f, -0.2f};
    upload(conv.bias().data_, b);

    std::vector<float> in(N * H * W);
    for (int i = 0; i < N * H * W; ++i) in[i] = 0.1f * (i + 1);
    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload(x, in);

    int64_t outH = 2, outW = 2;
    std::vector<float> grad_out(N * Cout * outH * outW);
    for (size_t i = 0; i < grad_out.size(); ++i) {
        grad_out[i] = 0.01f * static_cast<float>((i % 11) + 1);
    }
    Tensor go = Tensor::empty({N, Cout, outH, outW}, Dtype::Float32, Device::cuda(0));
    upload(go, grad_out);

    std::vector<float> ref_gi, ref_gw, ref_gb;
    ref_conv2d_backward(in, w, b.data(), grad_out, ref_gi, ref_gw, ref_gb,
                         N, Cin, H, W, Cout, kH, kW,
                         1, 1, 0, 0, 1, 1);

    auto grads = conv.backward(go, x);
    sync_stream();

    std::vector<float> gi(N * Cin * H * W);
    cudaMemcpy(gi.data(), grads.grad_input.data(),
               sizeof(float) * gi.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < gi.size(); ++i) {
        CHECK(gi[i] == doctest::Approx(ref_gi[i]).epsilon(1e-3));
    }

    std::vector<float> gw(Cout * Cin * kH * kW);
    cudaMemcpy(gw.data(), grads.grad_weight.data(),
               sizeof(float) * gw.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < gw.size(); ++i) {
        CHECK(gw[i] == doctest::Approx(ref_gw[i]).epsilon(1e-3));
    }

    std::vector<float> gb(Cout);
    cudaMemcpy(gb.data(), grads.grad_bias.data(),
               sizeof(float) * Cout, cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < Cout; ++i) {
        CHECK(gb[i] == doctest::Approx(ref_gb[i]).epsilon(1e-3));
    }
}

TEST_CASE("[gpu][fp32] conv2d_backward multi-channel Cin=2 Cout=3") {
    constexpr int64_t N = 1, Cin = 2, H = 4, W = 4;
    constexpr int64_t Cout = 3, kH = 3, kW = 3;
    Conv2d conv(Cin, Cout, kH, kW, 1, 1, 1, 1, 1, 1, /*bias*/true);

    std::vector<float> w(Cout * Cin * kH * kW);
    for (size_t i = 0; i < w.size(); ++i) w[i] = 0.02f * ((i % 7) + 1);
    upload(conv.weight().data_, w);
    std::vector<float> b = {0.3f, 0.1f, -0.2f};
    upload(conv.bias().data_, b);

    std::vector<float> in(N * Cin * H * W);
    for (size_t i = 0; i < in.size(); ++i) in[i] = 0.05f * ((i % 5) + 1);
    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload(x, in);

    int64_t outH = 4, outW = 4;
    std::vector<float> grad_out(N * Cout * outH * outW);
    for (size_t i = 0; i < grad_out.size(); ++i) {
        grad_out[i] = 0.01f * ((i % 13) + 1);
    }
    Tensor go = Tensor::empty({N, Cout, outH, outW}, Dtype::Float32, Device::cuda(0));
    upload(go, grad_out);

    std::vector<float> ref_gi, ref_gw, ref_gb;
    ref_conv2d_backward(in, w, b.data(), grad_out, ref_gi, ref_gw, ref_gb,
                         N, Cin, H, W, Cout, kH, kW,
                         1, 1, 1, 1, 1, 1);

    auto grads = conv.backward(go, x);
    sync_stream();

    CHECK(grads.grad_input.shape()[0] == N);
    CHECK(grads.grad_input.shape()[1] == Cin);
    CHECK(grads.grad_input.shape()[2] == H);
    CHECK(grads.grad_input.shape()[3] == W);
    CHECK(grads.grad_weight.shape()[0] == Cout);
    CHECK(grads.grad_weight.shape()[1] == Cin);
    CHECK(grads.grad_weight.shape()[2] == kH);
    CHECK(grads.grad_weight.shape()[3] == kW);
    CHECK(grads.grad_bias.shape()[0] == Cout);

    std::vector<float> gi(N * Cin * H * W);
    cudaMemcpy(gi.data(), grads.grad_input.data(),
               sizeof(float) * gi.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < gi.size(); ++i) {
        CHECK(gi[i] == doctest::Approx(ref_gi[i]).epsilon(1e-3));
    }

    std::vector<float> gw(Cout * Cin * kH * kW);
    cudaMemcpy(gw.data(), grads.grad_weight.data(),
               sizeof(float) * gw.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < gw.size(); ++i) {
        CHECK(gw[i] == doctest::Approx(ref_gw[i]).epsilon(1e-3));
    }

    std::vector<float> gb(Cout);
    cudaMemcpy(gb.data(), grads.grad_bias.data(),
               sizeof(float) * Cout, cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < Cout; ++i) {
        CHECK(gb[i] == doctest::Approx(ref_gb[i]).epsilon(1e-3));
    }
}

TEST_CASE("[gpu][fp32] conv2d_backward no-bias path returns zero grad_bias") {
    constexpr int64_t N = 1, Cin = 1, H = 3, W = 3;
    constexpr int64_t Cout = 1, kH = 2, kW = 2;
    Conv2d conv(Cin, Cout, kH, kW, 1, 1, 0, 0, 1, 1, /*bias*/false);

    std::vector<float> w = {0.5f, 0.5f, 0.5f, 0.5f};
    upload(conv.weight().data_, w);

    std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload(x, in);

    int64_t outH = 2, outW = 2;
    std::vector<float> grad_out(N * Cout * outH * outW);
    for (size_t i = 0; i < grad_out.size(); ++i) grad_out[i] = 0.1f * (i + 1);
    Tensor go = Tensor::empty({N, Cout, outH, outW}, Dtype::Float32, Device::cuda(0));
    upload(go, grad_out);

    std::vector<float> ref_gi, ref_gw, ref_gb;
    ref_conv2d_backward(in, w, nullptr, grad_out, ref_gi, ref_gw, ref_gb,
                         N, Cin, H, W, Cout, kH, kW,
                         1, 1, 0, 0, 1, 1);

    auto grads = conv.backward(go, x);
    sync_stream();

    std::vector<float> gi(N * Cin * H * W);
    cudaMemcpy(gi.data(), grads.grad_input.data(),
               sizeof(float) * gi.size(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < gi.size(); ++i) {
        CHECK(gi[i] == doctest::Approx(ref_gi[i]).epsilon(1e-3));
    }

    // grad_bias should be zeros (no bias was used in the forward pass).
    std::vector<float> gb(Cout);
    cudaMemcpy(gb.data(), grads.grad_bias.data(),
               sizeof(float) * Cout, cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < Cout; ++i) {
        CHECK(gb[i] == doctest::Approx(0.0f).epsilon(1e-6));
    }
}

TEST_CASE("[gpu][fp32] conv2d_backward shape validation") {
    constexpr int64_t N = 1, Cin = 2, H = 4, W = 4;
    constexpr int64_t Cout = 3, kH = 3, kW = 3;
    Conv2d conv(Cin, Cout, kH, kW, 1, 1, 1, 1, 1, 1, /*bias*/true);

    Tensor x = Tensor::empty({N, Cin, H, W}, Dtype::Float32, Device::cuda(0));
    upload(x, std::vector<float>(N * Cin * H * W, 0.1f));

    int64_t outH = 4, outW = 4;
    Tensor go = Tensor::empty({N, Cout, outH, outW}, Dtype::Float32, Device::cuda(0));
    upload(go, std::vector<float>(N * Cout * outH * outW, 0.1f));

    auto grads = conv.backward(go, x);
    sync_stream();

    CHECK(grads.grad_input.shape()[0] == N);
    CHECK(grads.grad_input.shape()[1] == Cin);
    CHECK(grads.grad_input.shape()[2] == H);
    CHECK(grads.grad_input.shape()[3] == W);
    CHECK(grads.grad_weight.shape()[0] == Cout);
    CHECK(grads.grad_weight.shape()[1] == Cin);
    CHECK(grads.grad_weight.shape()[2] == kH);
    CHECK(grads.grad_weight.shape()[3] == kW);
    CHECK(grads.grad_bias.shape()[0] == Cout);
    CHECK(grads.grad_input.numel() == N * Cin * H * W);
    CHECK(grads.grad_weight.numel() == Cout * Cin * kH * kW);
    CHECK(grads.grad_bias.numel() == Cout);
}
