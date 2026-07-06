#include "autograd/AddBackward.hpp"

namespace tensorforge {

std::vector<Tensor> AddBackward::apply(std::vector<Tensor>&& grads) {
    // z = a + b  →  dz/da = 1, dz/db = 1
    // Pass-through: both input gradients equal the output gradient.
    return {grads[0], grads[0]};
}

} // namespace tensorforge
