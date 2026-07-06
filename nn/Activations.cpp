#include "nn/Activations.hpp"

namespace tensorforge::nn {

Tensor ReLU::forward(Tensor x) {
    return x.relu();
}

Tensor Sigmoid::forward(Tensor x) {
    return x.sigmoid();
}

Tensor Tanh::forward(Tensor x) {
    return x.tanh();
}

} // namespace tensorforge::nn