#include "autograd/SavedTensor.hpp"

#include <stdexcept>

namespace tensorforge {

const Tensor& SavedTensor::unpack() const {
    if (data_.version() != version_at_save_) {
        throw std::runtime_error(
            "SavedTensor version mismatch: tensor was modified after being saved");
    }
    return data_;
}

} // namespace tensorforge
