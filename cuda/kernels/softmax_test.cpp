// TensorForge — softmax kernel tests (Wave 4 / T20)
//
// Smoke tests for launch_softmax covering:
//   - Single-row softmax sums to 1 (FP32)
//   - Known hand-computed case at uniform input gives 1/N
//   - Numerical stability on large-magnitude inputs (no NaN/Inf)
//   - FP16 single-row within 1% relative
//   - Multi-row (rows=4, cols=8) verifies per-row independence

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/softmax.cuh"
#include "tensor/Dtype.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using tensorforge::Device;
using tensorforge::DeviceContext;
using tensorforge::Dtype;
using tensorforge::Tensor;
using tensorforge::full;

namespace {

void sync_stream() {
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
}

// Upload a known FP32 host vector to a 2D [rows, cols] row-major tensor.
// FP16/BF16 paths cast on the host so the byte count matches the buffer.
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

TEST_CASE("softmax FP32 single row sums to 1") {
    constexpr int64_t kCols = 5;
    float host[kCols] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Tensor x = upload_fp32(host, 1, kCols);
    Tensor y = Tensor::empty({1, kCols}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_softmax(x.data(), y.data(), 1, kCols, Dtype::Float32,
                                (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[kCols];
    cudaMemcpy(out, y.data(), sizeof(float) * kCols, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < kCols; ++i) {
        sum += out[i];
    }
    CHECK(sum == doctest::Approx(1.0f).epsilon(1e-6));
    // Each entry must be in (0, 1).
    for (int i = 0; i < kCols; ++i) {
        CHECK(out[i] > 0.0f);
        CHECK(out[i] < 1.0f);
    }
}

TEST_CASE("softmax FP32 uniform input gives 1/N each") {
    constexpr int64_t kCols = 8;
    // All zeros -> exp(0)/N = 1/N.
    Tensor x = full({1, kCols}, 0.0f, Dtype::Float32, Device::cuda(0));
    Tensor y = Tensor::empty({1, kCols}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_softmax(x.data(), y.data(), 1, kCols, Dtype::Float32,
                                (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[kCols];
    cudaMemcpy(out, y.data(), sizeof(float) * kCols, cudaMemcpyDeviceToHost);
    const float expected = 1.0f / static_cast<float>(kCols);
    for (int i = 0; i < kCols; ++i) {
        CHECK(out[i] == doctest::Approx(expected).epsilon(1e-6));
    }
}

TEST_CASE("softmax FP32 numerical stability: large-magnitude inputs") {
    constexpr int64_t kCols = 4;
    // 1000 will overflow naive exp() but subtract-max stabilizes it.
    float host[kCols] = {1000.0f, 1001.0f, 1002.0f, 1003.0f};
    Tensor x = upload_fp32(host, 1, kCols);
    Tensor y = Tensor::empty({1, kCols}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_softmax(x.data(), y.data(), 1, kCols, Dtype::Float32,
                                (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[kCols];
    cudaMemcpy(out, y.data(), sizeof(float) * kCols, cudaMemcpyDeviceToHost);

    // No NaN, no Inf.
    for (int i = 0; i < kCols; ++i) {
        CHECK(std::isfinite(out[i]));
    }
    // Reference: expected probabilities for [0, 1, 2, 3] (after subtract max).
    const float exp_ref[4] = {
        std::exp(0.0f - 3.0f), std::exp(1.0f - 3.0f),
        std::exp(2.0f - 3.0f), std::exp(3.0f - 3.0f),
    };
    float sum = exp_ref[0] + exp_ref[1] + exp_ref[2] + exp_ref[3];
    for (int i = 0; i < kCols; ++i) {
        CHECK(out[i] == doctest::Approx(exp_ref[i] / sum).epsilon(1e-5));
    }
}

TEST_CASE("softmax FP32 multi-row independence") {
    constexpr int64_t kRows = 3;
    constexpr int64_t kCols = 4;

    // Each row has independent magnitudes; the softmax should NOT bleed
    // across rows.
    float host[kRows * kCols] = {
        0.0f, 0.0f, 0.0f, 0.0f,  // row 0: uniform -> 0.25 each
        1.0f, 2.0f, 3.0f, 4.0f,  // row 1: shifted-spike
        -1.0f, -1.0f, 5.0f, -1.0f,  // row 2: spike at index 2
    };
    Tensor x = upload_fp32(host, kRows, kCols);
    Tensor y = Tensor::empty({kRows, kCols}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_softmax(x.data(), y.data(), kRows, kCols, Dtype::Float32,
                                (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[kRows * kCols];
    cudaMemcpy(out, y.data(), sizeof(float) * kRows * kCols,
               cudaMemcpyDeviceToHost);

    // Row 0: 1/4 each.
    for (int i = 0; i < kCols; ++i) {
        CHECK(out[i] == doctest::Approx(0.25f).epsilon(1e-6));
    }

    // Row 1 and row 2 sums must each be 1 (independent).
    float r1_sum = 0.0f, r2_sum = 0.0f;
    for (int i = 0; i < kCols; ++i) {
        r1_sum += out[kCols + i];
        r2_sum += out[2 * kCols + i];
    }
    CHECK(r1_sum == doctest::Approx(1.0f).epsilon(1e-6));
    CHECK(r2_sum == doctest::Approx(1.0f).epsilon(1e-6));

    // Row 2 dominant at index 2.
    int64_t r2_max_idx = 0;
    float r2_max = out[2 * kCols];
    for (int i = 1; i < kCols; ++i) {
        if (out[2 * kCols + i] > r2_max) {
            r2_max = out[2 * kCols + i];
            r2_max_idx = i;
        }
    }
    CHECK(r2_max_idx == 2);
}

TEST_CASE("softmax FP16 within 1% of FP32 reference") {
    constexpr int64_t kCols = 8;
    float host[kCols] = {0.5f, 1.5f, 2.5f, 3.5f, -1.0f, 0.0f, 4.0f, 2.0f};
    Tensor x = upload_fp32(host, 1, kCols, Dtype::Float16);
    Tensor y = Tensor::empty({1, kCols}, Dtype::Float16, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_softmax(x.data(), y.data(), 1, kCols, Dtype::Float16,
                                (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    __half out[kCols];
    cudaMemcpy(out, y.data(), sizeof(__half) * kCols, cudaMemcpyDeviceToHost);

    // Reference: compute on host in FP32 then cast to FP16 for comparison.
    float maxv = host[0];
    for (int i = 1; i < kCols; ++i) maxv = std::max(maxv, host[i]);
    float sum = 0.0f;
    for (int i = 0; i < kCols; ++i) sum += std::exp(host[i] - maxv);
    float inv = 1.0f / sum;
    for (int i = 0; i < kCols; ++i) {
        float ref = std::exp(host[i] - maxv) * inv;
        CHECK(__half2float(out[i]) == doctest::Approx(ref).epsilon(0.05));
    }
}
