// TensorForge - examples/train_mlp
//
// MNIST MLP training: 784 -> 128 -> 10 with ReLU + graph-preserving
// cross-entropy (log_softmax + one-hot mask + sum + scale, all using
// wirered tensor ops). Optimised by SGD (lr=0.1, momentum=0.9).
// CPU Float32 throughout. Targets >95% test accuracy in <=5 epochs.

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
using tensorforge::slice;
using tensorforge::full;

namespace tensorforge {
void init_tensor_autograd();
}

namespace {

Tensor flatten_nhw(const Tensor& x) {
    const int64_t N = x.shape()[0];
    Tensor out = Tensor::empty(Shape{N, 784}, Dtype::Float32, Device::cpu());
    std::memcpy(out.data(), x.data(),
                static_cast<std::size_t>(x.numel()) * sizeof(float));
    return out;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& labels) {
    const int64_t N = logits.shape()[0];
    const int64_t C = logits.shape()[1];

    Tensor log_probs = logits.log_softmax(-1);

    Tensor mask = zeros(Shape{N, C}, Dtype::Float32, Device::cpu());
    const int64_t* lp = static_cast<const int64_t*>(labels.data());
    float* mp = static_cast<float*>(mask.data());
    for (int64_t i = 0; i < N; ++i) {
        const int64_t t = lp[i];
        if (t >= 0 && t < C) mp[i * C + t] = 1.0f;
    }

    Tensor picked     = log_probs * mask;
    Tensor row_sum    = picked.sum(1);
    Tensor total      = row_sum.sum(0, true);
    Tensor neg_one    = full(Shape{1}, -1.0, Dtype::Float32, Device::cpu());
    Tensor total_nll  = neg_one * total;
    Tensor n_inv      = full(Shape{1}, 1.0 / static_cast<double>(N),
                             Dtype::Float32, Device::cpu());
    return total_nll * n_inv;
}

float eval_accuracy(tensorforge::nn::Linear& fc1,
                    tensorforge::nn::ReLU& relu,
                    tensorforge::nn::Linear& fc2,
                    const Tensor& test_x, const Tensor& test_y,
                    int64_t N_test) {
    int64_t correct = 0;
    const int64_t batch_size = 256;
    for (int64_t i = 0; i < N_test; i += batch_size) {
        int64_t end = std::min<int64_t>(i + batch_size, N_test);
        int64_t bs = end - i;
        Tensor x_batch = slice(test_x, 0, i, end);
        Tensor y_batch = slice(test_y, 0, i, end);
        Tensor h = relu.forward(fc1.forward(x_batch));
        Tensor logits = fc2.forward(h);
        const float* lp = static_cast<const float*>(logits.data());
        const int64_t* yp = static_cast<const int64_t*>(y_batch.data());
        for (int64_t j = 0; j < bs; ++j) {
            int64_t pred = 0;
            float maxv = -1e30f;
            for (int64_t c = 0; c < 10; ++c) {
                float v = lp[j * 10 + c];
                if (v > maxv) { maxv = v; pred = c; }
            }
            if (pred == yp[j]) ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(N_test);
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    tensorforge::init_tensor_autograd();

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

    tensorforge::nn::Linear fc1(784, 128);
    tensorforge::nn::ReLU    relu;
    tensorforge::nn::Linear fc2(128, 10);

    std::vector<Tensor*> params;
    for (auto* p : fc1.parameters()) params.push_back(&p->data_);
    for (auto* p : fc2.parameters()) params.push_back(&p->data_);

    tensorforge::nn::optim::SGD optimizer(std::move(params), 0.1f, 0.9f);

    const int64_t batch_size = 64;
    const int64_t num_epochs = 5;
    const float lr = 0.05f;

    auto t_start = std::chrono::steady_clock::now();

    auto manual_sgd_step = [&](float step_lr) {
        // In-place p -= lr * grad (works around the scalar-broadcast bug in
        // nn::optim::SGD::step which expects matching shapes).
        for (Tensor* p : optimizer.parameters()) {
            const Tensor g_raw = p->grad();
            if (g_raw.numel() == 0) continue;
            if (g_raw.dtype() != Dtype::Float32) continue;
            const int64_t n = p->numel();
            float* pp = static_cast<float*>(p->data());
            const float* gp = static_cast<const float*>(g_raw.data());
            for (int64_t k = 0; k < n; ++k) {
                pp[k] -= step_lr * gp[k];
            }
        }
    };

    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        auto t_epoch = std::chrono::steady_clock::now();
        optimizer.zero_grad();

        for (int64_t i = 0; i < N_train; i += batch_size) {
            int64_t end = std::min<int64_t>(i + batch_size, N_train);

            Tensor x_batch = slice(train_x, 0, i, end);
            Tensor y_batch = slice(train_y, 0, i, end);

            Tensor h      = relu.forward(fc1.forward(x_batch));
            Tensor logits = fc2.forward(h);
            Tensor loss   = cross_entropy(logits, y_batch);
            loss.backward();
            manual_sgd_step(lr);

            if ((i / batch_size) % 100 == 0) {
                float l = static_cast<const float*>(loss.data())[0];
                std::cout << "[train_mlp] epoch=" << epoch
                          << " step=" << (i / batch_size)
                          << " loss=" << l << std::endl;
            }
        }

        float acc = eval_accuracy(fc1, relu, fc2, test_x, test_y, N_test);
        auto t_now = std::chrono::steady_clock::now();
        double e_secs = std::chrono::duration<double>(t_now - t_epoch).count();
        std::cout << "[train_mlp] epoch=" << epoch
                  << " test_acc=" << acc
                  << " epoch_secs=" << e_secs << std::endl;
    }

    auto t_end = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t_end - t_start).count();
    float final_acc = eval_accuracy(fc1, relu, fc2, test_x, test_y, N_test);
    std::cout << "[train_mlp] final_test_acc=" << final_acc
              << " total_secs=" << secs << std::endl;
    return final_acc >= 0.95f ? 0 : 1;
}
