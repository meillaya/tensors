// TensorForge - examples/train_mlp
//
// MNIST MLP training: 784 -> 256 -> 10 with ReLU + graph-preserving
// cross-entropy (log_softmax + one-hot mask + sum + scale, all using
// wirered tensor ops). Optimised by SGD (lr=0.05, momentum=0.9).
// CPU Float32 throughout.
//
// Usage:
//   bazel-bin/examples/train_mlp

#include "nn/Activations.hpp"
#include "nn/Linear.hpp"
#include "nn/optim/SGD.hpp"

#include "data/MnistLoader.hpp"

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"
#include "tensor/slicing.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using tensorforge::Dtype;
using tensorforge::Device;
using tensorforge::Shape;
using tensorforge::Tensor;
using tensorforge::data::load_mnist_test;
using tensorforge::data::load_mnist_train;
using tensorforge::zeros;

namespace {

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

void dbg(const char* tag, const Tensor& t) {
    std::cerr << "[DBG " << tag << "] req=" << t.requires_grad()
              << " shape.ndim=" << t.shape().ndim()
              << " numel=" << t.numel() << "\n";
}

Tensor diff_cross_entropy(const Tensor& logits, const Tensor& labels) {
    if (logits.dtype() != Dtype::Float32 || logits.shape().ndim() != 2) {
        throw std::invalid_argument("diff_cross_entropy: bad logits");
    }
    const int64_t N = logits.shape()[0];
    const int64_t C = logits.shape()[1];
    dbg("logits", logits);

    Tensor log_probs = logits.log_softmax(-1);
    dbg("log_probs", log_probs);

    Tensor mask = zeros(Shape{N, C}, Dtype::Float32, Device::cpu());
    const int64_t* lp = static_cast<const int64_t*>(labels.data());
    float* mp = static_cast<float*>(mask.data());
    for (int64_t i = 0; i < N; ++i) {
        const int64_t t = lp[i];
        if (t >= 0 && t < C) mp[i * C + t] = 1.0f;
    }
    dbg("mask", mask);

    Tensor picked     = log_probs * mask;
    dbg("picked", picked);
    Tensor row_sum    = picked.sum(1);
    dbg("row_sum", row_sum);
    Tensor total      = row_sum.sum(0, true);
    dbg("total", total);
    Tensor neg_one    = tensorforge::full(Shape{1}, -1.0,
                                          Dtype::Float32, Device::cpu());
    Tensor total_nll  = neg_one * total;
    dbg("total_nll", total_nll);
    Tensor n_inv      = tensorforge::full(
        Shape{1}, 1.0 / static_cast<double>(N),
        Dtype::Float32, Device::cpu());
    Tensor loss       = total_nll * n_inv;
    dbg("loss", loss);
    return loss;
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    std::string data_dir = "/data/mnist";
    if (const char* env = std::getenv("TENSORFORGE_MNIST_DIR")) {
        data_dir = env;
    }

    std::cout << "[train_mlp] data dir: " << data_dir << std::endl;
    auto [train_x_raw, train_y] = load_mnist_train(data_dir);
    auto [test_x_raw,  test_y ] = load_mnist_test(data_dir);

    const int64_t N_train = train_x_raw.shape()[0];
    const int64_t N_test  = test_x_raw.shape()[0];
    std::cout << "[train_mlp] train=" << N_train
              << " test=" << N_test << std::endl;

    Tensor train_x = flatten_nhw(train_x_raw);
    Tensor test_x  = flatten_nhw(test_x_raw);

    tensorforge::nn::Linear fc1(784, 256);
    tensorforge::nn::ReLU    relu;
    tensorforge::nn::Linear fc2(256, 10);

    std::vector<Tensor*> params;
    for (auto* p : fc1.parameters()) params.push_back(&p->data_);
    for (auto* p : fc2.parameters()) params.push_back(&p->data_);

    tensorforge::nn::optim::SGD optimizer(std::move(params), 0.05f, 0.9f);

    const int64_t batch_size = 64;
    const int64_t num_epochs = 1;

    auto t_start = std::chrono::steady_clock::now();

    int64_t step_count = 0;
    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();

        for (int64_t i = 0; i < N_train; i += batch_size) {
            const int64_t end = std::min(i + batch_size, N_train);

            Tensor x_batch = tensorforge::slice(train_x, 0, i, end);
            Tensor y_batch = tensorforge::slice(train_y, 0, i, end);

            Tensor h      = relu.forward(fc1.forward(x_batch));
            Tensor logits = fc2.forward(h);
            if (step_count == 0) {
                Tensor loss = diff_cross_entropy(logits, y_batch);
                loss.backward();
                optimizer.step();
                step_count++;
                break;
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "[train_mlp] done in " << secs << " s" << std::endl;
    return 0;
}
