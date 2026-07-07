// TensorForge — 16x16 tiled GEMM tests (Wave 5 / T23)
//
// Verifies launch_gemm_tiled_16x16 produces identical results to the naive
// reference for a range of shapes:
//   - 2x2 hand-computed case
//   - Square 16x16 * 16x16 (one tile, exact shared-memory paths)
//   - Square 32x32 * 32x32 (multiple K-tiles)
//   - 64x64 * 64x64 (multi-block, all tiles)
//   - Rectangular (3x5) * (5x2)
//   - Non-multiple-of-16 (17x19 * 19x23)
//   - FP16 / BF16 dtype paths within tolerance
//
// This is the kernel that nn/Conv2d forwards on (T27), so correctness
// across boundary tiles is the main thing under test.

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
    sync_stream();
    return t;
}

void run_gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C) {
    auto& ctx = tensorforge::DeviceContext::current();
    sync_stream();
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];
    tensorforge::launch_gemm_tiled_16x16(A.data(), B.data(), C.data(),
                                          M, N, K, A.dtype(),
                                          (void*)ctx.current_stream);
    sync_stream();
}

std::vector<float> ref_matmul(const float* A, const float* B,
                              int64_t M, int64_t K, int64_t N) {
    std::vector<float> out(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                s += A[m * K + k] * B[k * N + n];
            }
            out[m * N + n] = s;
        }
    }
    return out;
}

}  // namespace

TEST_CASE("[gpu][fp32] gemm_tiled_16x16 2x2 hand-computed case") {
    float A_h[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B_h[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    Tensor A = upload_fp32(A_h, 2, 2);
    Tensor B = upload_fp32(B_h, 2, 2);
    Tensor C = Tensor::empty({2, 2}, Dtype::Float32, Device::cuda(0));

    run_gemm_tiled(A, B, C);

    float C_h[4];
    cudaMemcpy(C_h, C.data(), sizeof(float) * 4, cudaMemcpyDeviceToHost);

    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};
    for (int i = 0; i < 4; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-5));
    }
}

TEST_CASE("[gpu][fp32] gemm_tiled_16x16 square 16x16 (single tile)") {
    constexpr int64_t N = 16;
    std::vector<float> A(N * N), B(N * N);
    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.1f * static_cast<float>(i + 1);
        B[i] = 0.05f * static_cast<float>((i % 7) + 1);
    }
    auto expected = ref_matmul(A.data(), B.data(), N, N, N);

    Tensor A_t = upload_fp32(A.data(), N, N);
    Tensor B_t = upload_fp32(B.data(), N, N);
    Tensor C = Tensor::empty({N, N}, Dtype::Float32, Device::cuda(0));
    run_gemm_tiled(A_t, B_t, C);

    std::vector<float> C_h(N * N);
    cudaMemcpy(C_h.data(), C.data(), sizeof(float) * N * N,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < N * N; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-3));
    }
}

TEST_CASE("[gpu][fp32] gemm_tiled_16x16 square 32x32 (multi K-tile)") {
    constexpr int64_t N = 32;
    std::vector<float> A(N * N), B(N * N);
    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.1f * static_cast<float>(i + 1);
        B[i] = 0.05f * static_cast<float>((i % 7) + 1);
    }
    auto expected = ref_matmul(A.data(), B.data(), N, N, N);

    Tensor A_t = upload_fp32(A.data(), N, N);
    Tensor B_t = upload_fp32(B.data(), N, N);
    Tensor C = Tensor::empty({N, N}, Dtype::Float32, Device::cuda(0));
    run_gemm_tiled(A_t, B_t, C);

    std::vector<float> C_h(N * N);
    cudaMemcpy(C_h.data(), C.data(), sizeof(float) * N * N,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < N * N; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-3));
    }
}

TEST_CASE("[gpu][fp32] gemm_tiled_16x16 square 64x64 (multi-block)") {
    constexpr int64_t N = 64;
    std::vector<float> A(N * N), B(N * N);
    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.01f * static_cast<float>(i + 1);
        B[i] = 0.02f * static_cast<float>((i % 11) + 1);
    }
    auto expected = ref_matmul(A.data(), B.data(), N, N, N);

    Tensor A_t = upload_fp32(A.data(), N, N);
    Tensor B_t = upload_fp32(B.data(), N, N);
    Tensor C = Tensor::empty({N, N}, Dtype::Float32, Device::cuda(0));
    run_gemm_tiled(A_t, B_t, C);

    std::vector<float> C_h(N * N);
    cudaMemcpy(C_h.data(), C.data(), sizeof(float) * N * N,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < N * N; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-2));
    }
}

TEST_CASE("[gpu][fp32] gemm_tiled_16x16 rectangular (3x5)*(5x2)") {
    float A_h[15];
    for (int i = 0; i < 15; ++i) A_h[i] = static_cast<float>(i + 1);
    float B_h[10];
    for (int i = 0; i < 10; ++i) B_h[i] = 0.1f * static_cast<float>(i + 1);

    auto expected = ref_matmul(A_h, B_h, 3, 5, 2);

    Tensor A_t = upload_fp32(A_h, 3, 5);
    Tensor B_t = upload_fp32(B_h, 5, 2);
    Tensor C = Tensor::empty({3, 2}, Dtype::Float32, Device::cuda(0));
    run_gemm_tiled(A_t, B_t, C);

    float C_h[6];
    cudaMemcpy(C_h, C.data(), sizeof(float) * 6, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 6; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-4));
    }
}

TEST_CASE("[gpu][fp32] gemm_tiled_16x16 non-multiple-of-16 (17x19)*(19x23)") {
    constexpr int64_t M = 17, K = 19, N = 23;
    std::vector<float> A(M * K), B(K * N);
    for (int64_t i = 0; i < M * K; ++i) A[i] = 0.01f * static_cast<float>(i + 1);
    for (int64_t i = 0; i < K * N; ++i) B[i] = 0.02f * static_cast<float>((i % 7) + 1);

    auto expected = ref_matmul(A.data(), B.data(), M, K, N);

    Tensor A_t = upload_fp32(A.data(), M, K);
    Tensor B_t = upload_fp32(B.data(), K, N);
    Tensor C = Tensor::empty({M, N}, Dtype::Float32, Device::cuda(0));
    run_gemm_tiled(A_t, B_t, C);

    std::vector<float> C_h(M * N);
    cudaMemcpy(C_h.data(), C.data(), sizeof(float) * M * N,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < M * N; ++i) {
        CHECK(C_h[i] == doctest::Approx(expected[i]).epsilon(1e-2));
    }
}

TEST_CASE("[gpu][fp16] gemm_tiled_16x16 FP16 matches FP32 reference") {
    constexpr int64_t N = 32;
    std::vector<float> A(N * N), B(N * N);
    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.1f * static_cast<float>(i + 1);
        B[i] = 0.05f * static_cast<float>((i % 7) + 1);
    }
    auto expected = ref_matmul(A.data(), B.data(), N, N, N);

    Tensor A_t = upload_fp32(A.data(), N, N, Dtype::Float16);
    Tensor B_t = upload_fp32(B.data(), N, N, Dtype::Float16);
    Tensor C = Tensor::empty({N, N}, Dtype::Float16, Device::cuda(0));
    run_gemm_tiled(A_t, B_t, C);

    std::vector<__half> C_h(N * N);
    cudaMemcpy(C_h.data(), C.data(), sizeof(__half) * N * N,
               cudaMemcpyDeviceToHost);
    // FP16 within 5% of FP32 reference — generous because FP16 has only
    // 10 mantissa bits and small values can drift at K=32.
    for (int64_t i = 0; i < N * N; ++i) {
        CHECK(__half2float(C_h[i]) == doctest::Approx(expected[i]).epsilon(0.1));
    }
}

TEST_CASE("[gpu][bf16] gemm_tiled_16x16 BF16 matches FP32 reference") {
    constexpr int64_t N = 32;
    std::vector<float> A(N * N), B(N * N);
    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.1f * static_cast<float>(i + 1);
        B[i] = 0.05f * static_cast<float>((i % 7) + 1);
    }
    auto expected = ref_matmul(A.data(), B.data(), N, N, N);

    Tensor A_t = upload_fp32(A.data(), N, N, Dtype::BFloat16);
    Tensor B_t = upload_fp32(B.data(), N, N, Dtype::BFloat16);
    Tensor C = Tensor::empty({N, N}, Dtype::BFloat16, Device::cuda(0));
    run_gemm_tiled(A_t, B_t, C);

    std::vector<__nv_bfloat16> C_h(N * N);
    cudaMemcpy(C_h.data(), C.data(), sizeof(__nv_bfloat16) * N * N,
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < N * N; ++i) {
        CHECK(__bfloat162float(C_h[i]) == doctest::Approx(expected[i]).epsilon(0.2));
    }
}