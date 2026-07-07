// TensorForge — naive GEMM tests (Wave 5 / T22)
//
// Smoke tests for launch_gemm_naive covering:
//   - 2x2 hand-computed known case ([1,2;3,4] * [5,6;7,8] = [19,22;43,50])
//   - Identity left-multiplication
//   - Non-square shapes (rectangular)
//   - Non-multiples-of-16 (bounds-checked path)
//   - FP16 / BF16 dtypes within tolerance of the FP32 reference
//   - Zero / one matrices edge cases

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/gemm.cuh"
#include "tensor/Dtype.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using tensorforge::Device;
using tensorforge::DeviceContext;
using tensorforge::Dtype;
using tensorforge::Tensor;
using tensorforge::full;

namespace {

// Synchronize the per-thread default stream so subsequent D2H copies see
// committed writes from the GEMM kernel.
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
    sync_stream();
    return t;
}

// Allocate a 2D output tensor and D2H copy into the supplied host buffer.
void download_fp32(const Tensor& t, float* host) {
    int64_t n = t.numel();
    cudaMemcpy(host, t.data(), n * sizeof(float), cudaMemcpyDeviceToHost);
}

void run_gemm(const Tensor& A, const Tensor& B, Tensor& C) {
    auto& ctx = tensorforge::DeviceContext::current();
    sync_stream();
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];
    tensorforge::launch_gemm_naive(A.data(), B.data(), C.data(),
                                   M, N, K, A.dtype(),
                                   (void*)ctx.current_stream);
    sync_stream();
}

}  // namespace

TEST_CASE("[gpu][fp32] gemm_naive 2x2 hand-computed case") {
    float A_h[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B_h[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};

    Tensor A = upload_fp32(A_h, 2, 2);
    Tensor B = upload_fp32(B_h, 2, 2);
    Tensor C = Tensor::empty({2, 2}, Dtype::Float32, Device::cuda(0));

    run_gemm(A, B, C);

    float C_h[4];
    download_fp32(C, C_h);
    for (int i = 0; i < 4; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-5));
    }
}

TEST_CASE("[gpu][fp32] gemm_naive identity preserves A") {
    // 4x4 identity
    float I_h[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    float A_h[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };

    Tensor A = upload_fp32(A_h, 4, 4);
    Tensor I = upload_fp32(I_h, 4, 4);
    Tensor C = Tensor::empty({4, 4}, Dtype::Float32, Device::cuda(0));

    run_gemm(A, I, C);

    float C_h[16];
    download_fp32(C, C_h);
    for (int i = 0; i < 16; ++i) {
        CHECK(C_h[i] == doctest::Approx(A_h[i]).epsilon(1e-5));
    }
}

TEST_CASE("[gpu][fp32] gemm_naive rectangular (3x5) * (5x2)") {
    float A_h[15];
    for (int i = 0; i < 15; ++i) A_h[i] = static_cast<float>(i + 1);
    float B_h[10];
    for (int i = 0; i < 10; ++i) B_h[i] = 0.1f * static_cast<float>(i + 1);

    // Reference: matrix multiply on host.
    float expected[6] = {0.0f};
    for (int m = 0; m < 3; ++m) {
        for (int n = 0; n < 2; ++n) {
            float s = 0.0f;
            for (int k = 0; k < 5; ++k) {
                s += A_h[m * 5 + k] * B_h[k * 2 + n];
            }
            expected[m * 2 + n] = s;
        }
    }

    Tensor A = upload_fp32(A_h, 3, 5);
    Tensor B = upload_fp32(B_h, 5, 2);
    Tensor C = Tensor::empty({3, 2}, Dtype::Float32, Device::cuda(0));

    run_gemm(A, B, C);

    float C_h[6];
    download_fp32(C, C_h);
    for (int i = 0; i < 6; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-4));
    }
}

TEST_CASE("[gpu][fp32] gemm_naive non-multiple-of-16 bounds") {
    // 17x19 * 19x23 — forces the boundary tile path.
    constexpr int64_t M = 17, K = 19, N = 23;
    std::vector<float> A_h(M * K), B_h(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_h[i] = 0.01f * static_cast<float>(i + 1);
    for (int64_t i = 0; i < K * N; ++i) B_h[i] = 0.02f * static_cast<float>((i % 7) + 1);

    // Reference: host multiply.
    std::vector<float> expected(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                s += A_h[m * K + k] * B_h[k * N + n];
            }
            expected[m * N + n] = s;
        }
    }

    Tensor A = upload_fp32(A_h.data(), M, K);
    Tensor B = upload_fp32(B_h.data(), K, N);
    Tensor C = Tensor::empty({M, N}, Dtype::Float32, Device::cuda(0));

    run_gemm(A, B, C);

    std::vector<float> C_h(M * N);
    download_fp32(C, C_h.data());
    for (int64_t i = 0; i < M * N; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-3));
    }
}

TEST_CASE("[gpu][fp32] gemm_naive zeros yields zeros") {
    constexpr int64_t M = 4, K = 4, N = 4;
    std::vector<float> zero(M * K, 0.0f);
    std::vector<float> B_h(K * N, 0.0f);
    for (int64_t i = 0; i < K * N; ++i) B_h[i] = static_cast<float>(i + 1);

    Tensor A = upload_fp32(zero.data(), M, K);
    Tensor B = upload_fp32(B_h.data(), K, N);
    Tensor C = Tensor::empty({M, N}, Dtype::Float32, Device::cuda(0));

    run_gemm(A, B, C);

    std::vector<float> C_h(M * N);
    download_fp32(C, C_h.data());
    for (int64_t i = 0; i < M * N; ++i) {
        CHECK(C_h[i] == doctest::Approx(0.0f).epsilon(1e-7));
    }
}

TEST_CASE("[gpu][fp16] gemm_naive FP16 within tolerance of FP32 reference") {
    float A_h[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B_h[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    Tensor A = upload_fp32(A_h, 2, 2, Dtype::Float16);
    Tensor B = upload_fp32(B_h, 2, 2, Dtype::Float16);
    Tensor C = Tensor::empty({2, 2}, Dtype::Float16, Device::cuda(0));

    run_gemm(A, B, C);

    __half C_h[4];
    cudaMemcpy(C_h, C.data(), sizeof(__half) * 4, cudaMemcpyDeviceToHost);
    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    for (int i = 0; i < 4; ++i) {
        CHECK(__half2float(C_h[i]) == doctest::Approx(expected[i]).epsilon(0.05));
    }
}

TEST_CASE("[gpu][bf16] gemm_naive BF16 within tolerance of FP32 reference") {
    float A_h[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B_h[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    Tensor A = upload_fp32(A_h, 2, 2, Dtype::BFloat16);
    Tensor B = upload_fp32(B_h, 2, 2, Dtype::BFloat16);
    Tensor C = Tensor::empty({2, 2}, Dtype::BFloat16, Device::cuda(0));

    run_gemm(A, B, C);

    __nv_bfloat16 C_h[4];
    cudaMemcpy(C_h, C.data(), sizeof(__nv_bfloat16) * 4, cudaMemcpyDeviceToHost);
    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    for (int i = 0; i < 4; ++i) {
        CHECK(__bfloat162float(C_h[i]) == doctest::Approx(expected[i]).epsilon(0.1));
    }
}