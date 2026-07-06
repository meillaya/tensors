#pragma once
#include "tensor/Tensor.hpp"

#include <string>
#include <utility>

namespace tensorforge::data {

// Load CIFAR-10 from binary batch files
// data_dir should contain data_batch_1..5 + test_batch
std::pair<Tensor, Tensor> load_cifar10_train(const std::string& data_dir);
std::pair<Tensor, Tensor> load_cifar10_test(const std::string& data_dir);

// Returns (images, labels) where:
//   images: (N, 3, 32, 32) Float32, NCHW layout, normalized per-channel
//   labels: (N,) Int64, class indices 0-9
} // namespace tensorforge::data
