// TensorForge - examples/train_cnn
//
// CIFAR-10 CNN training scaffold (T47):
//   Conv2d(3,16,3) -> ReLU -> Conv2d(16,32,3,stride=2) -> ReLU
//     -> Linear(32*7*7, 128) -> ReLU -> Linear(128, 10)
//
// NOTE (T47 partial): Training will likely crash with
//   "backward() called on tensor without grad_fn"
// due to the known autograd requires_grad propagation bug (same as
// T45/t45). This file exists as a scaffold; the bug is documented
// separately. We verify the example builds cleanly so the wiring
// path is exercised by CI.
//
// v1 uses Conv2dModule (CPU forward with Kaiming init). MaxPool is
// approximated by stride=2 downsample. Build:
//   bazelisk build //examples:train_cnn

#include "nn/Activations.hpp"
#include "nn/Conv2dModule.hpp"
#include "nn/CrossEntropyLoss.hpp"
#include "nn/Linear.hpp"
#include "nn/optim/SGD.hpp"

#include "data/CifarLoader.hpp"

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"
#include "tensor/slicing.hpp"
#include "tensor/shape_ops.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using tensorforge::Dtype;
using tensorforge::Device;
using tensorforge::Shape;
using tensorforge::Tensor;
using tensorforge::data::load_cifar10_train;
using tensorforge::data::load_cifar10_test;

int main(int /*argc*/, char** /*argv*/) {
    std::string data_dir = "/data/cifar";
    if (const char* env = std::getenv("TENSORFORGE_CIFAR_DIR")) {
        data_dir = env;
    }

    std::cout << "[train_cnn] data dir: " << data_dir << std::endl;

    auto [train_x, train_y] = load_cifar10_train(data_dir);
    auto [test_x,  test_y ] = load_cifar10_test(data_dir);

    const int64_t N_train = train_x.shape()[0];
    const int64_t N_test  = test_x.shape()[0];
    std::cout << "[train_cnn] train=" << N_train
              << " test=" << N_test << std::endl;

    // Architecture (v1, CPU forward): stride=2 downsample stands in
    // for the missing MaxPool2d module.
    tensorforge::nn::Conv2dModule conv1(3, 16, 3);
    tensorforge::nn::ReLU         relu;
    tensorforge::nn::Conv2dModule conv2(16, 32, 3, /*stride=*/2);
    tensorforge::nn::Linear       fc1(32 * 7 * 7, 128);
    tensorforge::nn::Linear       fc2(128, 10);
    tensorforge::nn::CrossEntropyLoss cel;

    std::vector<Tensor*> params;
    for (auto* p : conv1.parameters()) params.push_back(&p->data_);
    for (auto* p : conv2.parameters()) params.push_back(&p->data_);
    for (auto* p : fc1.parameters())   params.push_back(&p->data_);
    for (auto* p : fc2.parameters())   params.push_back(&p->data_);

    tensorforge::nn::optim::SGD optimizer(std::move(params), 0.01f, 0.9f);

    const int64_t batch_size = 32;
    const int64_t num_epochs = 2;
    const int64_t max_samples = std::min<int64_t>(N_train, 1000);  // time budget

    for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();

        for (int64_t i = 0; i < max_samples; i += batch_size) {
            const int64_t end = std::min(i + batch_size, max_samples);
            const int64_t bs  = end - i;

            Tensor x_batch = tensorforge::slice(train_x, 0, i, end);
            Tensor y_batch = tensorforge::slice(train_y, 0, i, end);

            Tensor h   = relu.forward(conv1.forward(x_batch));
            Tensor h2  = relu.forward(conv2.forward(h));
            Tensor flat = tensorforge::reshape(h2, Shape{bs, 32 * 7 * 7});
            Tensor h3  = relu.forward(fc1.forward(flat));
            Tensor logits = fc2.forward(h3);
            Tensor loss = cel.forward(logits, y_batch);

            // Likely fails with autograd requires_grad propagation bug
            // (documented alongside T45 / T47). Kept for explicit intent.
            loss.backward();
            optimizer.step();

            if (i % 200 == 0) {
                const float* lp = static_cast<const float*>(loss.data());
                std::cout << "Epoch " << epoch
                          << " step " << i
                          << " loss=" << lp[0] << std::endl;
            }
        }
        std::cout << "[train_cnn] Epoch " << epoch << " done" << std::endl;
    }

    std::cout << "[train_cnn] done" << std::endl;
    return 0;
}
