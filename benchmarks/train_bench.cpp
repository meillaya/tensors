// TensorForge - benchmarks/train_bench (T49)
//
// Training throughput benchmark. Times forward-only of an MLP on MNIST
// (784 -> 256 -> 10) using ReLU + cross-entropy. Backward is omitted
// because of the known autograd requires_grad propagation bug (same as
// T45/T47 partials). All ops use CPU Float32 - nn/Linear, nn/ReLU,
// nn/CrossEntropyLoss are CPU-only in v1.
//
// Build:
//   bazelisk run //benchmarks:train_bench -- \
//       --output=/data/tensorforge/benchmarks/results/train_bench_results.json
//
// Output: JSON array of {model, mode, sec_per_epoch, samples_per_sec}.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "nn/Activations.hpp"
#include "nn/CrossEntropyLoss.hpp"
#include "nn/Linear.hpp"

#include "data/MnistLoader.hpp"

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"
#include "tensor/shape_ops.hpp"
#include "tensor/slicing.hpp"

using tensorforge::Dtype;
using tensorforge::Device;
using tensorforge::Shape;
using tensorforge::Tensor;
using tensorforge::data::load_mnist_train;
using tensorforge::nn::CrossEntropyLoss;
using tensorforge::nn::Linear;
using tensorforge::nn::ReLU;

namespace {

double time_fn_us(std::function<void()> fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() / iters * 1e6;
}

// Flatten (N, 1, 28, 28) -> (N, 784) on CPU.
Tensor flatten_nhw(const Tensor& x) {
    if (x.dtype() != Dtype::Float32) {
        throw std::invalid_argument("flatten_nhw: expected Float32");
    }
    const int64_t N = x.shape()[0];
    Tensor out = Tensor::empty(Shape{N, 784}, Dtype::Float32, Device::cpu());
    std::memcpy(out.data(), x.data(),
                static_cast<std::size_t>(x.numel()) * sizeof(float));
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string output_path =
        "/data/tensorforge/benchmarks/results/train_bench_results.json";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        const std::string prefix = "--output=";
        if (a.rfind(prefix, 0) == 0) {
            output_path = a.substr(prefix.size());
        } else if (a == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    std::string data_dir = "/data/mnist";
    if (const char* env = std::getenv("TENSORFORGE_MNIST_DIR")) {
        data_dir = env;
    }
    std::cout << "[train_bench] data dir: " << data_dir << std::endl;

    auto [train_x_raw, train_y] = load_mnist_train(data_dir);
    const int64_t N_full = train_x_raw.shape()[0];
    std::cout << "[train_bench] full train N=" << N_full << std::endl;

    // Flatten images to (N, 784) then take a small subset for the time
    // budget. Subset stays on CPU; Linear/ReLU/CEL are CPU-only in v1.
    Tensor train_x_flat = flatten_nhw(train_x_raw);

    const int64_t N = 1000;  // small subset, fits in the 30-min budget
    Tensor x = tensorforge::slice(train_x_flat, 0, 0, N);
    Tensor y = tensorforge::slice(train_y, 0, 0, N);

    std::cout << "[train_bench] x shape=[";
    for (size_t i = 0; i < x.shape().ndim(); ++i) {
        std::cout << x.shape()[i] << (i + 1 == x.shape().ndim() ? "" : ",");
    }
    std::cout << "] dtype=" << (x.dtype() == Dtype::Float32 ? "f32" : "?")
              << " device=" << (x.device() == Device::cpu() ? "cpu" : "cuda")
              << std::endl;

    // Model - CPU Float32.
    Linear fc1(784, 256);
    ReLU   relu;
    Linear fc2(256, 10);
    CrossEntropyLoss cel;

    const int64_t batch_size = 64;
    const int64_t num_batches = N / batch_size;
    std::cout << "[train_bench] batch=" << batch_size
              << " num_batches=" << num_batches << std::endl;

    // Forward-only timing (per-batch microseconds).
    double forward_us_total = 0.0;
    int64_t measured_batches = 0;
    for (int64_t b = 0; b < num_batches; ++b) {
        const int64_t start = b * batch_size;
        const int64_t end   = start + batch_size;
        Tensor xb = tensorforge::slice(x, 0, start, end);
        Tensor yb = tensorforge::slice(y, 0, start, end);

        auto step = [&]() {
            Tensor h      = relu.forward(fc1.forward(xb));
            Tensor logits = fc2.forward(h);
            Tensor loss   = cel.forward(logits, yb);
            // Prevent the compiler from eliding the whole forward.
            std::atomic_thread_fence(std::memory_order_release);
            (void)loss.numel();
        };
        // Warmup once (covers Linear/ReLU/CEL dispatcher paths), then
        // time with low iters to fit the budget.
        const double t_us = time_fn_us(step, /*warmup=*/1, /*iters=*/3);
        forward_us_total += t_us;
        ++measured_batches;
        std::cout << "[train_bench] batch " << b << " forward=" << t_us
                  << " us" << std::endl;
    }

    const double sec_per_epoch = forward_us_total * 1e-6;
    const double measured_samples =
        static_cast<double>(batch_size * measured_batches);
    const double samples_per_sec =
        measured_samples / std::max(sec_per_epoch, 1e-9);

    std::cout << "[train_bench] sec_per_epoch=" << sec_per_epoch
              << " samples_per_sec=" << samples_per_sec << std::endl;

    // Write JSON.
    std::ofstream f(output_path);
    if (!f) {
        std::cerr << "[train_bench] cannot open output: " << output_path
                  << std::endl;
        return 1;
    }
    f << "[\n";
    f << "  {\"model\": \"mlp_mnist\", \"mode\": \"forward_only\", "
      << "\"sec_per_epoch\": " << sec_per_epoch
      << ", \"samples_per_sec\": " << samples_per_sec << "},\n";
    f << "  {\"model\": \"mlp_mnist\", \"mode\": \"forward_only\", "
      << "\"batch_size\": " << batch_size
      << ", \"num_batches\": " << num_batches
      << ", \"samples\": " << measured_samples << "}\n";
    f << "]\n";
    f.close();

    std::cout << "[train_bench] wrote " << output_path << std::endl;
    return 0;
}
