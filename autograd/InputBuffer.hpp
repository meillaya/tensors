#pragma once

#include "tensor/Tensor.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace tensorforge {

// InputBuffer — accumulates output gradients for a single Node from
// multiple paths in the backward graph. When a Node is ready (all
// upstream nodes have been processed), the engine calls take() to
// extract the accumulated gradients and passes them to Node::apply().
class InputBuffer {
public:
    InputBuffer() = default;
    explicit InputBuffer(size_t num_outputs);

    void add(size_t pos, Tensor grad);

    std::vector<Tensor> take();

private:
    std::vector<std::optional<Tensor>> buffer_;
};

} // namespace tensorforge
