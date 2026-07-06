#pragma once
#include "tensor/Tensor.hpp"

#include <string>
#include <utility>

namespace tensorforge::data {

// Load MNIST from local IDX files
// data_dir should contain train-images-idx3-ubyte, train-labels-idx1-ubyte,
// t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
std::pair<Tensor, Tensor> load_mnist_train(const std::string& data_dir);
std::pair<Tensor, Tensor> load_mnist_test(const std::string& data_dir);

// Returns (images, labels) where:
//   images: (N, 1, 28, 28) Float32, normalized
//   labels: (N,) Int64, class indices 0-9
} // namespace tensorforge::data
