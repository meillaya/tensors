// TensorForge — GEMM benchmark (Wave 5 / T25)
//
// Times naive, 16x16-tiled, optimized, and cuBLAS GEMM kernels on the H100
// pod and emits a JSON report. Sizes are squared (N x N x N) at
// N = {128, 256, 512, 1024, 2048}. Each kernel runs `warmup` warmup launches
// and `iters` timed launches; reported time is the median.
//
// JSON shape:
//   {
//     "gpu": "<name from cudaGetDeviceProperties>",
//     "dtype": "float32" | "float16" | "bfloat16",
//     "sizes": [N1, N2, ...],
//     "results": [
//       {"size": N, "kernel": "naive|tiled|optimized|cublas",
//        "time_ms": ..., "tflops": ..., "sol_pct": ...},
//       ...
//     ]
//   }
//
// `--output=path.json` writes the report to `path.json`. Default is stdout.
//
// Build: `bazelisk run //benchmarks:gemm_bench -- --output=/data/benchmarks/gemm_results.json`

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
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

constexpr int kWarmup = 3;
constexpr int kIters = 10;
constexpr int kSizes[] = {128, 256, 512, 1024, 2048};
constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

// H100 PCIe FP32 peak: 51 TFLOPS; A100 40GB FP32: 19.5 TFLOPS. Use a
// reasonable mid-point that the H100 pod should easily beat for the
// optimized kernels.
constexpr double kPeakTflopsH100 = 51.0;
constexpr double kPeakTflopsA100 = 19.5;

// The H100 pod is the reference target. We autodetect by SM count.
double peak_tflops(int sm_count) {
    // sm_90 -> H100, sm_80 -> A100. (The H100 has 132 SMs, the A100 has
    // 108 SMs; threshold at 120 keeps us on the safe side.)
    return (sm_count >= 120) ? kPeakTflopsH100 : kPeakTflopsA100;
}

const char* gpu_name(int sm_count) {
    return (sm_count >= 120) ? "H100" : "A100";
}

// cuBLAS handle (one per process; reused across all sizes / dtypes).
cublasHandle_t cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
        cudaStream_t stream =
            reinterpret_cast<cudaStream_t>(DeviceContext::current().current_stream);
        cublasSetStream(handle, stream);
    }
    return handle;
}

struct BenchResult {
    int size;
    std::string kernel;
    double time_ms;
    double tflops;
    double sol_pct;
};

double bench_with_event(const std::function<void()>& launch) {
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    for (int i = 0; i < kWarmup; ++i) launch();
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times;
    times.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        cudaEventRecord(start, stream);
        launch();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(times.begin(), times.end());
    return static_cast<double>(times[times.size() / 2]);
}

// 2 * M * N * K is the standard "useful FLOPs" count for C = A @ B in FP
// matmul (multiply + add per inner-product element).
double tflops_for(int M, int N, int K, double ms) {
    double flops = 2.0 * static_cast<double>(M) * N * K;
    return flops / (ms * 1e-3) / 1e12;
}

// Run a TensorForge GEMM and return median ms.
double bench_tensorforge_gemm(const Tensor& A, const Tensor& B, Tensor& C,
                              Dtype dtype,
                              void (*launcher)(const void*, const void*, void*,
                                               int64_t, int64_t, int64_t,
                                               Dtype, void*)) {
    auto& ctx = tensorforge::DeviceContext::current();
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    auto run = [&]() {
        launcher(A.data(), B.data(), C.data(), M, N, K, dtype, (void*)stream);
    };
    return bench_with_event(run);
}

// cuBLAS FP32 GEMM: row-major C = A @ B. cuBLAS is column-major, so we
// compute (B^T @ A^T)^T by swapping op order. C^T = B^T @ A^T means in
// column-major view we want cublasSgemm with A_cm = B_row^T, B_cm =
// A_row^T, then C_cm = C_row^T. Equivalently: C_row = A_row @ B_row
// becomes cublasGemmEx(transA=N, transB=N, m=N, n=M, k=K, A=B, lda=N,
// B=A, ldb=K, B=C, ldc=N). The "B=" naming is confusing but it's just
// passing buffers in the column-major interpretation.
double bench_cublas_fp32(const Tensor& A, const Tensor& B, Tensor& C) {
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cublasSetStream(cublas_handle(), stream);
    auto run = [&]() {
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major C(M,N) = A(M,K) @ B(K,N). Use cuBLAS as if we wanted
        // C_cm(N,M) = B_cm(N,K) @ A_cm(K,M), i.e. m=N, n=M, k=K.
        cublasSgemm(cublas_handle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha,
                    static_cast<const float*>(B.data()), static_cast<int>(N),
                    static_cast<const float*>(A.data()), static_cast<int>(K),
                    &beta,
                    static_cast<float*>(C.data()), static_cast<int>(N));
    };
    return bench_with_event(run);
}

double bench_cublas_fp16(const Tensor& A, const Tensor& B, Tensor& C) {
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cublasSetStream(cublas_handle(), stream);
    auto run = [&]() {
        const __half alpha = __float2half(1.0f);
        const __half beta = __float2half(0.0f);
        cublasHgemm(cublas_handle(),
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha,
                    reinterpret_cast<const __half*>(B.data()), static_cast<int>(N),
                    reinterpret_cast<const __half*>(A.data()), static_cast<int>(K),
                    &beta,
                    reinterpret_cast<__half*>(C.data()), static_cast<int>(N));
    };
    return bench_with_event(run);
}

double bench_cublas_bf16(const Tensor& A, const Tensor& B, Tensor& C) {
    int64_t M = A.shape()[0];
    int64_t K = A.shape()[1];
    int64_t N = B.shape()[1];
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cublasSetStream(cublas_handle(), stream);
    auto run = [&]() {
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle(),
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                     &alpha,
                     B.data(), CUDA_R_16BF, static_cast<int>(N),
                     A.data(), CUDA_R_16BF, static_cast<int>(K),
                     &beta,
                     C.data(), CUDA_R_16BF, static_cast<int>(N),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    return bench_with_event(run);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') { out += "\\n"; }
        else { out.push_back(c); }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string output_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--output=", 0) == 0) {
            output_path = arg.substr(9);
        }
    }

    auto& ctx = tensorforge::DeviceContext::current();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    double peak = peak_tflops(sm_count);
    const char* gpu = gpu_name(sm_count);

    std::cerr << "[gemm_bench] GPU=" << gpu << " (SMs=" << sm_count
              << "), peak FP32=" << peak << " TFLOPS\n";

    std::vector<BenchResult> results;

    for (Dtype dtype : {Dtype::Float32, Dtype::Float16, Dtype::BFloat16}) {
        const char* dtype_name =
            (dtype == Dtype::Float32) ? "float32" :
            (dtype == Dtype::Float16) ? "float16" : "bfloat16";

        std::cerr << "[gemm_bench] === dtype=" << dtype_name << " ===\n";

        for (int idx = 0; idx < kNumSizes; ++idx) {
            int64_t N = kSizes[idx];

            // Allocate buffers as the target dtype.
            Tensor A = full({N, N}, 0.5f, dtype, Device::cuda(0));
            Tensor B = full({N, N}, 0.3f, dtype, Device::cuda(0));
            Tensor C = Tensor::empty({N, N}, dtype, Device::cuda(0));
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

            // naive
            double t_naive = bench_tensorforge_gemm(A, B, C, dtype,
                tensorforge::launch_gemm_naive);
            double fl = tflops_for(N, N, N, t_naive);
            results.push_back({static_cast<int>(N), "naive", t_naive, fl,
                                100.0 * fl / peak});

            // tiled
            double t_tiled = bench_tensorforge_gemm(A, B, C, dtype,
                tensorforge::launch_gemm_tiled_16x16);
            double fl_t = tflops_for(N, N, N, t_tiled);
            results.push_back({static_cast<int>(N), "tiled", t_tiled, fl_t,
                                100.0 * fl_t / peak});

            // optimized
            double t_opt = bench_tensorforge_gemm(A, B, C, dtype,
                tensorforge::launch_gemm_optimized);
            double fl_o = tflops_for(N, N, N, t_opt);
            results.push_back({static_cast<int>(N), "optimized", t_opt, fl_o,
                                100.0 * fl_o / peak});

            // cuBLAS (fp32 / fp16 / bf16 each have their own call).
            double t_cb = 0.0;
            if (dtype == Dtype::Float32) {
                t_cb = bench_cublas_fp32(A, B, C);
            } else if (dtype == Dtype::Float16) {
                t_cb = bench_cublas_fp16(A, B, C);
            } else {
                t_cb = bench_cublas_bf16(A, B, C);
            }
            double fl_c = tflops_for(N, N, N, t_cb);
            results.push_back({static_cast<int>(N), "cublas", t_cb, fl_c,
                                100.0 * fl_c / peak});

            std::cerr << "[gemm_bench]   N=" << N
                      << "  naive=" << t_naive << " ms (" << fl << " TFLOPS)"
                      << "  tiled=" << t_tiled << " ms (" << fl_t << " TFLOPS)"
                      << "  opt=" << t_opt << " ms (" << fl_o << " TFLOPS)"
                      << "  cublas=" << t_cb << " ms (" << fl_c << " TFLOPS)"
                      << "\n";
        }
    }

    // ---- Emit JSON ----
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"gpu\": \"" << json_escape(gpu) << "\",\n";
    ss << "  \"sm_count\": " << sm_count << ",\n";
    ss << "  \"peak_tflops\": " << peak << ",\n";
    ss << "  \"warmup\": " << kWarmup << ",\n";
    ss << "  \"iters\": " << kIters << ",\n";
    ss << "  \"sizes\": [";
    for (int i = 0; i < kNumSizes; ++i) {
        if (i > 0) ss << ", ";
        ss << kSizes[i];
    }
    ss << "],\n";
    ss << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ss << "    {\"size\": " << r.size
           << ", \"kernel\": \"" << r.kernel << "\""
           << ", \"time_ms\": " << r.time_ms
           << ", \"tflops\": " << r.tflops
           << ", \"sol_pct\": " << r.sol_pct
           << "}";
        if (i + 1 < results.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";

    std::string json = ss.str();
    if (!output_path.empty()) {
        std::ofstream out(output_path);
        if (!out) {
            std::cerr << "[gemm_bench] Failed to open " << output_path << "\n";
            return 1;
        }
        out << json;
        std::cerr << "[gemm_bench] Wrote " << output_path << " ("
                  << json.size() << " bytes, " << results.size() << " entries)\n";
    } else {
        std::cout << json;
    }
    return 0;
}