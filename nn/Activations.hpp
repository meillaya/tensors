#pragma once

#include "nn/Module.hpp"
#include "tensor/Tensor.hpp"

namespace tensorforge::nn {

// ReLU: y = max(x, 0).
// v1: forward delegates to Tensor::relu(). Real elementwise + autograd
// wiring for the backward pass lands with T19 / T33-T35; in v1 these
// modules are forward-only smoke tests.
class ReLU : public Module {
public:
    Tensor forward(Tensor x) override;
};

// Sigmoid: y = 1 / (1 + exp(-x)).
class Sigmoid : public Module {
public:
    Tensor forward(Tensor x) override;
};

// Tanh: y = tanh(x).
class Tanh : public Module {
public:
    Tensor forward(Tensor x) override;
};

} // namespace tensorforge::nn