// TensorForge - Conv2dModule implementation (Wave 7 / T45)
//
// Construction:
//   * weight ~ U(-bound, bound) where bound = sqrt(1 / (Cin * k * k))
//     (Kaiming uniform, fan_in mode - matches nn.Linear's init).
//   * bias = 0 (or absent if bias == false).
//
// forward(): CPU reference 2D conv with stride / padding / dilation.

#include "nn/Conv2dModule.hpp"

#include "tensor/Dtype.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace tensorforge::nn {

namespace {

void cpu_conv2d(const float* in, const float* weight,
                const float* bias, float* out,
                int64_t N, int64_t Cin, int64_t H, int64_t W,
                int64_t Cout, int64_t kH, int64_t kW,
                int64_t stride, int64_t padding, int64_t dilation,
                int64_t outH, int64_t outW) {
    const int64_t in_stride_n = Cin * H * W;
    const int64_t in_stride_c = H * W;
    const int64_t w_stride_co = Cin * kH * kW;
    const int64_t w_stride_ci = kH * kW;
    const int64_t out_stride_n = Cout * outH * outW;
    const int64_t out_stride_c = outH * outW;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t co = 0; co < Cout; ++co) {
            for (int64_t oh = 0; oh < outH; ++oh) {
                for (int64_t ow = 0; ow < outW; ++ow) {
                    float acc = 0.0f;
                    for (int64_t ci = 0; ci < Cin; ++ci) {
                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                const int64_t ih = oh * stride - padding + kh * dilation;
                                const int64_t iw = ow * stride - padding + kw * dilation;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                    continue;
                                }
                                const float xv = in[n * in_stride_n
                                                    + ci * in_stride_c
                                                    + ih * W + iw];
                                const float wgt = weight[co * w_stride_co
                                                         + ci * w_stride_ci
                                                         + kh * kW + kw];
                                acc += xv * wgt;
                            }
                        }
                    }
                    if (bias) {
                        acc += bias[co];
                    }
                    out[n * out_stride_n + co * out_stride_c + oh * outW + ow] = acc;
                }
            }
        }
    }
}

}  // namespace

Conv2dModule::Conv2dModule(int in_channels, int out_channels, int kernel_size,
                            int stride, int padding, int dilation,
                            bool bias)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      bias_enabled_(bias) {
    Tensor w = Tensor::empty(
        Shape{out_channels, in_channels, kernel_size, kernel_size},
        Dtype::Float32, Device::cpu());

    const float bound = std::sqrt(1.0f / static_cast<float>(
        in_channels * kernel_size * kernel_size));
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-bound, bound);

    const int64_t n_elem = w.numel();
    float* w_data = static_cast<float*>(w.data());
    for (int64_t i = 0; i < n_elem; ++i) {
        w_data[i] = dist(rng);
    }
    w.requires_grad(true);
    register_parameter("weight", Parameter(std::move(w)));

    if (bias_enabled_) {
        Tensor b = zeros(Shape{out_channels}, Dtype::Float32, Device::cpu());
        b.requires_grad(true);
        register_parameter("bias", Parameter(std::move(b)));
    }
}

Tensor Conv2dModule::forward(Tensor x) {
    const auto& in_shape = x.shape();
    if (in_shape.ndim() != 4) {
        throw std::invalid_argument("Conv2dModule.forward expects [N, Cin, H, W]");
    }
    const int64_t N  = in_shape[0];
    const int64_t Cin = in_shape[1];
    const int64_t H = in_shape[2];
    const int64_t W = in_shape[3];
    if (Cin != in_channels_) {
        throw std::invalid_argument("Conv2dModule.forward: Cin mismatch");
    }
    if (x.dtype() != Dtype::Float32) {
        throw std::invalid_argument("Conv2dModule.forward: only Float32 supported");
    }
    if (x.device().type != DeviceType::CPU) {
        throw std::invalid_argument("Conv2dModule.forward: only CPU supported in v1");
    }
    if (kernel_size_ <= 0 || stride_ <= 0 || dilation_ <= 0) {
        throw std::invalid_argument("Conv2dModule.forward: non-positive stride/dilation/kernel");
    }

    const int64_t kH = kernel_size_;
    const int64_t kW = kernel_size_;
    const int64_t eff_kH = dilation_ * (kH - 1) + 1;
    const int64_t eff_kW = dilation_ * (kW - 1) + 1;
    const int64_t outH = (H + 2 * padding_ - eff_kH) / stride_ + 1;
    const int64_t outW = (W + 2 * padding_ - eff_kW) / stride_ + 1;

    Tensor out = zeros(Shape{N, out_channels_, outH, outW},
                       Dtype::Float32, Device::cpu());

    const float* w_ptr = static_cast<const float*>(parameters_.at("weight").data_.data());
    const float* b_ptr = bias_enabled_
        ? static_cast<const float*>(parameters_.at("bias").data_.data())
        : nullptr;

    cpu_conv2d(static_cast<const float*>(x.data()),
               w_ptr, b_ptr,
               static_cast<float*>(out.data()),
               N, Cin, H, W,
               out_channels_, kH, kW,
               stride_, padding_, dilation_,
               outH, outW);

    return out;
}

}  // namespace tensorforge::nn
