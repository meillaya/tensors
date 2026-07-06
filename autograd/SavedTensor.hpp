#pragma once

#include "tensor/Tensor.hpp"

#include <cstdint>

namespace tensorforge {

// SavedTensor — snapshot of a forward-pass tensor plus its version counter.
// unpack() errors if the underlying storage was mutated after saving,
// catching in-place modifications that would invalidate the backward pass.
class SavedTensor {
public:
    Tensor data_;
    uint32_t version_at_save_;

    explicit SavedTensor(Tensor t)
        : data_(std::move(t)), version_at_save_(data_.version()) {}

    // Unpack with version check; throws std::runtime_error on mismatch.
    const Tensor& unpack() const;
};

} // namespace tensorforge
