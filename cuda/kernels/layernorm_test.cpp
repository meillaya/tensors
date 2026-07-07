// TensorForge — layer-norm kernel tests (Wave 4 / T21)
//
// Smoke tests for launch_layernorm covering:
//   - Constant row input -> all zeros after normalize (gamma/beta affine
//     shift only)
//   - Random single row: mean ≈ 0, var ≈ 1 after normalization (before
//     gamma/beta), and gamma/beta affine applies correctly
//   - Multi-row independence (rows=4, cols=8)
//   - FP16 multi-row within tolerance of FP32 reference

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/layernorm.cuh"
#include "tensor/Dtype.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using tensorforge::Device;
using tensorforge::DeviceContext;
using tensorforge::Dtype;
using tensorforge::Tensor;
using tensorforge::full;
using tensorforge::ones;
using tensorforge::zeros;

namespace {

void sync_stream() {
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
}

Tensor upload_fp32(const float* host, int64_t rows, int64_t cols,
                   Dtype dtype = Dtype::Float32) {
    Tensor t = Tensor::empty({rows, cols}, dtype, Device::cuda(0));
    int64_t n = rows * cols;
    cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(DeviceContext::current().current_stream);
    switch (dtype) {
    case Dtype::Float32: {
        cudaMemcpyAsync(t.data(), host, sizeof(float) * n,
                        cudaMemcpyHostToDevice, stream);
        break;
    }
    case Dtype::Float16: {
        std::vector<__half> casted(n);
        for (int64_t i = 0; i < n; ++i) casted[i] = __float2half(host[i]);
        cudaMemcpyAsync(t.data(), casted.data(), sizeof(__half) * n,
                        cudaMemcpyHostToDevice, stream);
        break;
    }
    case Dtype::BFloat16: {
        std::vector<__nv_bfloat16> casted(n);
        for (int64_t i = 0; i < n; ++i) {
            casted[i] = __float2bfloat16_rn(host[i]);
        }
        cudaMemcpyAsync(t.data(), casted.data(), sizeof(__nv_bfloat16) * n,
                        cudaMemcpyHostToDevice, stream);
        break;
    }
    default:
        throw std::invalid_argument("upload_fp32: unsupported dtype");
    }
    cudaStreamSynchronize(stream);
    return t;
}

}  // namespace

TEST_CASE("layernorm FP32 constant-row input: zero output with affine") {
    constexpr int64_t kCols = 4;
    // All 3.0 -> mean=3, var=0 -> (3-3)/sqrt(0+eps) * gamma + beta = beta.
    constexpr float v = 3.0f;
    Tensor x = full({1, kCols}, v, Dtype::Float32, Device::cuda(0));

    // gamma=2, beta=-1: y = 0*2 + -1 = -1 everywhere.
    float gamma_h[kCols] = {2.0f, 2.0f, 2.0f, 2.0f};
    float beta_h[kCols] = {-1.0f, -1.0f, -1.0f, -1.0f};
    Tensor gamma = upload_fp32(gamma_h, 1, kCols);
    Tensor beta = upload_fp32(beta_h, 1, kCols);

    Tensor y = Tensor::empty({1, kCols}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    constexpr float eps = 1e-5f;
    tensorforge::launch_layernorm(x.data(), gamma.data(), beta.data(), y.data(),
                                  1, kCols, eps, Dtype::Float32,
                                  (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[kCols];
    cudaMemcpy(out, y.data(), sizeof(float) * kCols, cudaMemcpyDeviceToHost);
    for (int i = 0; i < kCols; ++i) {
        CHECK(out[i] == doctest::Approx(-1.0f).epsilon(1e-4));
    }
}

TEST_CASE("layernorm FP32 single-row normalize then affine") {
    constexpr int64_t kCols = 8;
    // Hand-picked values where mean and variance are easy to compute.
    float host[kCols] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Tensor x = upload_fp32(host, 1, kCols);

    // gamma = ones, beta = zeros: pure normalize, expect mean≈0, var≈1.
    Tensor gamma = ones({kCols}, Dtype::Float32, Device::cuda(0));
    Tensor beta = zeros({kCols}, Dtype::Float32, Device::cuda(0));

    Tensor y = Tensor::empty({1, kCols}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    constexpr float eps = 1e-5f;
    tensorforge::launch_layernorm(x.data(), gamma.data(), beta.data(), y.data(),
                                  1, kCols, eps, Dtype::Float32,
                                  (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[kCols];
    cudaMemcpy(out, y.data(), sizeof(float) * kCols, cudaMemcpyDeviceToHost);

    // Compute reference: mean=4.5, var=5.25 (pop variance, /cols).
    float ref_mean = 0.0f;
    for (int i = 0; i < kCols; ++i) ref_mean += host[i];
    ref_mean /= static_cast<float>(kCols);
    float ref_var = 0.0f;
    for (int i = 0; i < kCols; ++i) {
        float d = host[i] - ref_mean;
        ref_var += d * d;
    }
    ref_var /= static_cast<float>(kCols);
    float inv_std = 1.0f / std::sqrt(ref_var + eps);

    float got_mean = 0.0f, got_var = 0.0f;
    for (int i = 0; i < kCols; ++i) got_mean += out[i];
    got_mean /= static_cast<float>(kCols);
    for (int i = 0; i < kCols; ++i) {
        float d = out[i] - got_mean;
        got_var += d * d;
    }
    got_var /= static_cast<float>(kCols);

    CHECK(got_mean == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(got_var == doctest::Approx(1.0f).epsilon(1e-3));
}

TEST_CASE("layernorm FP32 multi-row independence") {
    constexpr int64_t kRows = 3;
    constexpr int64_t kCols = 4;

    // Each row has independent stats.
    float host[kRows * kCols] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -2.0f, 1.0f, 5.0f, 0.0f,
        10.0f, 10.0f, 10.0f, 10.0f,  // row 2: constant -> output = beta.
    };
    Tensor x = upload_fp32(host, kRows, kCols);

    Tensor gamma = ones({kCols}, Dtype::Float32, Device::cuda(0));
    Tensor beta = zeros({kCols}, Dtype::Float32, Device::cuda(0));

    Tensor y = Tensor::empty({kRows, kCols}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    constexpr float eps = 1e-5f;
    tensorforge::launch_layernorm(x.data(), gamma.data(), beta.data(), y.data(),
                                  kRows, kCols, eps, Dtype::Float32,
                                  (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[kRows * kCols];
    cudaMemcpy(out, y.data(), sizeof(float) * kRows * kCols,
               cudaMemcpyDeviceToHost);

    // Row 0 and row 1 must have mean≈0 each.
    for (int r = 0; r < 2; ++r) {
        float mean = 0.0f;
        for (int c = 0; c < kCols; ++c) mean += out[r * kCols + c];
        mean /= static_cast<float>(kCols);
        CHECK(mean == doctest::Approx(0.0f).epsilon(1e-4));
    }
    // Row 2 (constant): output must be ~0 everywhere (pure normalize with
    // gamma=1, beta=0).
    for (int c = 0; c < kCols; ++c) {
        CHECK(out[2 * kCols + c] == doctest::Approx(0.0f).epsilon(1e-3));
    }
}

TEST_CASE("layernorm FP16 within tolerance of FP32 reference") {
    constexpr int64_t kRows = 2;
    constexpr int64_t kCols = 8;

    float host[kRows * kCols] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        -1.0f, 0.5f, -2.0f, 3.5f, 4.0f, -0.5f, 2.5f, 1.5f,
    };
    Tensor x = upload_fp32(host, kRows, kCols, Dtype::Float16);

    // Workaround: tensor/factory.cpp's fill_with_value encodes FP16 via
    // int16 cast, so full({...}, 1.0f, FP16, ...) is wrong. Upload directly.
    float gamma_f[kCols];
    float beta_f[kCols];
    for (int i = 0; i < kCols; ++i) {
        gamma_f[i] = 1.0f;
        beta_f[i] = 0.0f;
    }
    Tensor gamma = upload_fp32(gamma_f, 1, kCols, Dtype::Float16);
    Tensor beta = upload_fp32(beta_f, 1, kCols, Dtype::Float16);

    Tensor y = Tensor::empty({kRows, kCols}, Dtype::Float16, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    constexpr float eps = 1e-3f;
    tensorforge::launch_layernorm(x.data(), gamma.data(), beta.data(), y.data(),
                                  kRows, kCols, eps, Dtype::Float16,
                                  (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    __half out[kRows * kCols];
    cudaMemcpy(out, y.data(), sizeof(__half) * kRows * kCols,
               cudaMemcpyDeviceToHost);

    // Compare to a per-row FP32 reference (mean 0, var 1 in normalized
    // coords).
    constexpr float fp16_tol = 0.05f;
    for (int r = 0; r < kRows; ++r) {
        float ref_mean = 0.0f;
        for (int c = 0; c < kCols; ++c) ref_mean += host[r * kCols + c];
        ref_mean /= static_cast<float>(kCols);
        float ref_var = 0.0f;
        for (int c = 0; c < kCols; ++c) {
            float d = host[r * kCols + c] - ref_mean;
            ref_var += d * d;
        }
        ref_var /= static_cast<float>(kCols);
        float inv_std = 1.0f / std::sqrt(ref_var + eps);
        for (int c = 0; c < kCols; ++c) {
            float ref = (host[r * kCols + c] - ref_mean) * inv_std;
            CHECK(__half2float(out[r * kCols + c]) ==
                  doctest::Approx(ref).epsilon(fp16_tol));
        }
    }
}

TEST_CASE("layernorm FP32 with affine gamma/beta shifts correctly") {
    constexpr int64_t kCols = 3;
    float host[kCols] = {0.0f, 1.0f, -1.0f};
    Tensor x = upload_fp32(host, 1, kCols);

    float gamma_h[kCols] = {2.0f, 2.0f, 2.0f};
    float beta_h[kCols] = {1.0f, 1.0f, 1.0f};
    Tensor gamma = upload_fp32(gamma_h, 1, kCols);
    Tensor beta = upload_fp32(beta_h, 1, kCols);

    Tensor y = Tensor::empty({1, kCols}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    constexpr float eps = 1e-5f;
    tensorforge::launch_layernorm(x.data(), gamma.data(), beta.data(), y.data(),
                                  1, kCols, eps, Dtype::Float32,
                                  (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[kCols];
    cudaMemcpy(out, y.data(), sizeof(float) * kCols, cudaMemcpyDeviceToHost);

    // Reference: mean ≈ 0, std ≈ 1.0954, then * 2 + 1.
    const float mean = 0.0f;
    const float var = (0.0f * 0.0f + 1.0f * 1.0f + (-1.0f) * (-1.0f)) / 3.0f;
    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < kCols; ++c) {
        float ref = (host[c] - mean) * inv_std * 2.0f + 1.0f;
        CHECK(out[c] == doctest::Approx(ref).epsilon(1e-4));
    }
}
