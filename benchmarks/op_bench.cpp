// TensorForge — op-level benchmark harness (T48)
//
// Times every operator the runtime exposes on CUDA: add, mul, relu,
// sigmoid, tanh, matmul (optimized), softmax, layer-norm. Emits a JSON
// report to the path given by `--output=...` (default
// `/data/tensorforge/benchmarks/results/op_bench_results.json`).
//
// Timing methodology: cudaEvent_t around each iteration, taking the
// median over `kIters` runs after `kWarmup` warmup launches. Mirrors
// benchmarks/gemm_bench.cpp.
//
// Op-specific notes:
//   * Elementwise ops (add, mul, relu, sigmoid, tanh) use the public
//     Tensor API (operator+, .relu(), .sigmoid(), .tanh()). These
//     dispatch to the registered CUDA launchers from
//     cuda/kernels/elementwise.cu.
//   * matmul, softmax, layer_norm are CPU-only in v1 (Tensor.cpp
//     rejects DeviceType::CUDA), so the harness calls the low-level
//     launch_* functions directly with raw Tensor::data() pointers and
//     the per-device stream.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/elementwise.cuh"
#include "cuda/kernels/gemm.cuh"
#include "cuda/kernels/layernorm.cuh"
#include "cuda/kernels/softmax.cuh"
#include "tensor/Device.hpp"
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
constexpr int kIters = 20;
constexpr Dtype kDtype = Dtype::Float32;

// Vector sizes for elementwise / unary ops.
constexpr int64_t kVecSizes[] = {1024, 65536, 1 << 18, 1 << 22, 1 << 24};
constexpr int kNumVecSizes = sizeof(kVecSizes) / sizeof(kVecSizes[0]);

// Squared sizes for GEMM (M = N = K).
constexpr int kMatmulSizes[] = {128, 256, 512, 1024, 2048};
constexpr int kNumMatmulSizes = sizeof(kMatmulSizes) / sizeof(kMatmulSizes[0]);

// 2D (rows, cols) configs for softmax / layernorm.
struct RowCol {
    int64_t rows;
    int64_t cols;
};
constexpr RowCol kSoftmaxShapes[] = {
    {1, 1024}, {32, 1024}, {128, 1024}, {1024, 1024}, {4096, 1024}};
constexpr int kNumSoftmaxShapes = sizeof(kSoftmaxShapes) / sizeof(kSoftmaxShapes[0]);

constexpr RowCol kLayerNormShapes[] = {
    {1, 1024}, {32, 1024}, {128, 1024}, {1024, 1024}};
constexpr int kNumLayerNormShapes = sizeof(kLayerNormShapes) / sizeof(kLayerNormShapes[0]);

double median_ms(std::vector<float> samples) {
    std::sort(samples.begin(), samples.end());
    return static_cast<double>(samples[samples.size() / 2]);
}

double bench_event(const std::function<void()>& launch, cudaStream_t stream) {
    for (int i = 0; i < kWarmup; ++i) launch();
    cudaStreamSynchronize(stream);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        cudaEventRecord(start, stream);
        launch();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        samples.push_back(ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return median_ms(std::move(samples));
}

double gb_per_s(int64_t bytes, double ms) {
    return static_cast<double>(bytes) / (ms * 1e-3) / 1e9;
}

double tflops_for_gemm(int64_t M, int64_t N, int64_t K, double ms) {
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                   static_cast<double>(K);
    return flops / (ms * 1e-3) / 1e12;
}

std::string gpu_name(int sm_count) {
    return (sm_count >= 120) ? "H100" : "A100";
}

std::string js_str(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') { out += "\\n"; }
        else { out.push_back(c); }
    }
    out.push_back('"');
    return out;
}

std::string fmt(double v, int prec) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return std::string(buf);
}

// Build a JSON object from an alternating key/value vector. Values are
// emitted as-is — string values must already be JSON-quoted (use
// js_str), numeric values must be pre-stringified (use fmt).
std::string make_entry(const std::vector<std::string>& kv) {
    std::ostringstream ss;
    ss << "{";
    for (size_t i = 0; i < kv.size(); i += 2) {
        if (i > 0) ss << ", ";
        ss << "\"" << kv[i] << "\": " << kv[i + 1];
    }
    ss << "}";
    return ss.str();
}

// Convenience: elementwise bandwidth-bound entry.
std::string ew_entry(const std::string& op, int64_t n, double ms, double bw) {
    return make_entry({"op", js_str(op),
                       "size", std::to_string(n),
                       "us", fmt(ms * 1e3, 3),
                       "time_ms", fmt(ms, 4),
                       "bandwidth_gb_s", fmt(bw, 2)});
}

// Convenience: 2D row/col op entry (softmax / layernorm).
std::string rc_entry(const std::string& op, int64_t rows, int64_t cols,
                     double ms, double bw) {
    return make_entry({"op", js_str(op),
                       "rows", std::to_string(rows),
                       "cols", std::to_string(cols),
                       "size", std::to_string(rows * cols),
                       "us", fmt(ms * 1e3, 3),
                       "time_ms", fmt(ms, 4),
                       "bandwidth_gb_s", fmt(bw, 2)});
}

// Convenience: matmul entry.
std::string mm_entry(int64_t n, double ms, double tflops) {
    return make_entry({"op", js_str("matmul"),
                       "size", std::to_string(n),
                       "us", fmt(ms * 1e3, 3),
                       "time_ms", fmt(ms, 4),
                       "tflops", fmt(tflops, 2)});
}

}  // namespace

int main(int argc, char** argv) {
    std::string output_path = "/data/tensorforge/benchmarks/results/op_bench_results.json";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--output=", 0) == 0) {
            output_path = arg.substr(9);
        }
    }

    std::string mkdir_cmd = "mkdir -p ";
    if (auto slash = output_path.find_last_of('/'); slash != std::string::npos) {
        mkdir_cmd += output_path.substr(0, slash);
    }
    (void)std::system(mkdir_cmd.c_str());

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = ctx.current_stream;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    std::cerr << "[op_bench] GPU=" << gpu_name(sm_count) << " (SMs=" << sm_count
              << "), stream=" << (stream ? "non-null" : "null") << "\n";
    std::cerr << "[op_bench] warmup=" << kWarmup << " iters=" << kIters
              << " dtype=float32\n";

    std::vector<std::string> entries;

    // ---- Elementwise binary ops (add, mul) ----
    for (int idx = 0; idx < kNumVecSizes; ++idx) {
        int64_t n = kVecSizes[idx];
        Tensor a = full({n}, 0.5f, kDtype, Device::cuda(0));
        Tensor b = full({n}, 0.7f, kDtype, Device::cuda(0));
        Tensor out = Tensor::empty({n}, kDtype, Device::cuda(0));
        cudaStreamSynchronize(stream);

        auto run_add = [&]() {
            launch_add(a.data(), b.data(), out.data(), n, kDtype,
                       reinterpret_cast<void*>(stream));
        };
        double ms_add = bench_event(run_add, stream);
        int64_t bytes = 3 * n * 4;  // 2 reads + 1 write, fp32
        entries.push_back(ew_entry("add", n, ms_add, gb_per_s(bytes, ms_add)));
        std::cerr << "[op_bench] add n=" << n << "  us=" << ms_add * 1e3
                  << "  bw=" << gb_per_s(bytes, ms_add) << " GB/s\n";

        auto run_mul = [&]() {
            launch_mul(a.data(), b.data(), out.data(), n, kDtype,
                       reinterpret_cast<void*>(stream));
        };
        double ms_mul = bench_event(run_mul, stream);
        entries.push_back(ew_entry("mul", n, ms_mul, gb_per_s(bytes, ms_mul)));
        std::cerr << "[op_bench] mul n=" << n << "  us=" << ms_mul * 1e3
                  << "  bw=" << gb_per_s(bytes, ms_mul) << " GB/s\n";
    }

    // ---- Unary activations (relu, sigmoid, tanh) ----
    for (int idx = 0; idx < kNumVecSizes; ++idx) {
        int64_t n = kVecSizes[idx];
        Tensor x = full({n}, 0.3f, kDtype, Device::cuda(0));
        Tensor out = Tensor::empty({n}, kDtype, Device::cuda(0));
        cudaStreamSynchronize(stream);
        int64_t bytes = 2 * n * 4;  // 1 read + 1 write

        auto run_relu = [&]() {
            launch_relu(x.data(), out.data(), n, kDtype,
                        reinterpret_cast<void*>(stream));
        };
        double ms = bench_event(run_relu, stream);
        entries.push_back(ew_entry("relu", n, ms, gb_per_s(bytes, ms)));
        std::cerr << "[op_bench] relu n=" << n << "  us=" << ms * 1e3
                  << "  bw=" << gb_per_s(bytes, ms) << " GB/s\n";

        auto run_sig = [&]() {
            launch_sigmoid(x.data(), out.data(), n, kDtype,
                           reinterpret_cast<void*>(stream));
        };
        double ms_sig = bench_event(run_sig, stream);
        entries.push_back(ew_entry("sigmoid", n, ms_sig, gb_per_s(bytes, ms_sig)));
        std::cerr << "[op_bench] sigmoid n=" << n << "  us=" << ms_sig * 1e3
                  << "  bw=" << gb_per_s(bytes, ms_sig) << " GB/s\n";

        auto run_tanh = [&]() {
            launch_tanh(x.data(), out.data(), n, kDtype,
                        reinterpret_cast<void*>(stream));
        };
        double ms_tanh = bench_event(run_tanh, stream);
        entries.push_back(ew_entry("tanh", n, ms_tanh, gb_per_s(bytes, ms_tanh)));
        std::cerr << "[op_bench] tanh n=" << n << "  us=" << ms_tanh * 1e3
                  << "  bw=" << gb_per_s(bytes, ms_tanh) << " GB/s\n";
    }

    // ---- GEMM (optimized) ----
    for (int idx = 0; idx < kNumMatmulSizes; ++idx) {
        int64_t n = kMatmulSizes[idx];
        Tensor A = full({n, n}, 0.5f, kDtype, Device::cuda(0));
        Tensor B = full({n, n}, 0.3f, kDtype, Device::cuda(0));
        Tensor C = Tensor::empty({n, n}, kDtype, Device::cuda(0));
        cudaStreamSynchronize(stream);

        auto run_gemm = [&]() {
            launch_gemm_optimized(A.data(), B.data(), C.data(),
                                  n, n, n, kDtype,
                                  reinterpret_cast<void*>(stream));
        };
        double ms = bench_event(run_gemm, stream);
        double tf = tflops_for_gemm(n, n, n, ms);
        entries.push_back(mm_entry(n, ms, tf));
        std::cerr << "[op_bench] matmul n=" << n << "  us=" << ms * 1e3
                  << "  tflops=" << tf << "\n";
    }

    // ---- Softmax ----
    for (int idx = 0; idx < kNumSoftmaxShapes; ++idx) {
        int64_t rows = kSoftmaxShapes[idx].rows;
        int64_t cols = kSoftmaxShapes[idx].cols;
        Tensor x = full({rows, cols}, 0.1f, kDtype, Device::cuda(0));
        Tensor y = Tensor::empty({rows, cols}, kDtype, Device::cuda(0));
        cudaStreamSynchronize(stream);

        auto run_sm = [&]() {
            launch_softmax(x.data(), y.data(), rows, cols, kDtype,
                           reinterpret_cast<void*>(stream));
        };
        double ms = bench_event(run_sm, stream);
        int64_t bytes = 2 * rows * cols * 4;
        entries.push_back(rc_entry("softmax", rows, cols, ms, gb_per_s(bytes, ms)));
        std::cerr << "[op_bench] softmax rows=" << rows << " cols=" << cols
                  << "  us=" << ms * 1e3 << "  bw=" << gb_per_s(bytes, ms) << " GB/s\n";
    }

    // ---- LayerNorm ----
    for (int idx = 0; idx < kNumLayerNormShapes; ++idx) {
        int64_t rows = kLayerNormShapes[idx].rows;
        int64_t cols = kLayerNormShapes[idx].cols;
        Tensor x = full({rows, cols}, 0.4f, kDtype, Device::cuda(0));
        Tensor gamma = full({cols}, 1.0f, kDtype, Device::cuda(0));
        Tensor beta = full({cols}, 0.0f, kDtype, Device::cuda(0));
        Tensor y = Tensor::empty({rows, cols}, kDtype, Device::cuda(0));
        cudaStreamSynchronize(stream);

        constexpr float kEps = 1e-5f;
        auto run_ln = [&]() {
            launch_layernorm(x.data(), gamma.data(), beta.data(), y.data(),
                             rows, cols, kEps, kDtype,
                             reinterpret_cast<void*>(stream));
        };
        double ms = bench_event(run_ln, stream);
        // in + out + gamma(cols) + beta(cols) bytes
        int64_t bytes = (2 * rows * cols + 2 * cols) * 4;
        entries.push_back(rc_entry("layernorm", rows, cols, ms, gb_per_s(bytes, ms)));
        std::cerr << "[op_bench] layernorm rows=" << rows << " cols=" << cols
                  << "  us=" << ms * 1e3 << "  bw=" << gb_per_s(bytes, ms) << " GB/s\n";
    }

    // ---- Emit JSON ----
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"gpu\": " << js_str(gpu_name(sm_count)) << ",\n";
    ss << "  \"sm_count\": " << sm_count << ",\n";
    ss << "  \"dtype\": \"float32\",\n";
    ss << "  \"warmup\": " << kWarmup << ",\n";
    ss << "  \"iters\": " << kIters << ",\n";
    ss << "  \"entries\": [\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        ss << "    " << entries[i];
        if (i + 1 < entries.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";

    std::string json = ss.str();
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "[op_bench] Failed to open " << output_path << " for write\n";
        return 1;
    }
    out << json;
    out.close();
    std::cerr << "[op_bench] Wrote " << output_path << " (" << json.size()
              << " bytes, " << entries.size() << " entries)\n";

    return 0;
}