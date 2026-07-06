#include "tensor/Tensor.hpp"

#include "tensor/CPUStorageAllocator.hpp"

#include <stdexcept>

namespace tensorforge {

Tensor Tensor::empty(Shape shape, Dtype dtype, Device device) {
    if (device.type != DeviceType::CPU) {
        throw std::invalid_argument("Tensor::empty only supports CPU devices in Wave 2");
    }

    const int64_t n = shape.numel();
    const size_t size_bytes = static_cast<size_t>(n) * dtype_size(dtype);

    Storage storage = CPUStorageAllocator::instance().allocate(size_bytes, device, dtype);
    Stride stride = Stride::compute_row_major(shape);

    return Tensor(std::move(storage), 0, std::move(shape), std::move(stride));
}

Tensor Tensor::relu() const {
    // STUB — see T19.
    return *this;
}

Tensor Tensor::sigmoid() const {
    // STUB — see T19.
    return *this;
}

Tensor Tensor::tanh() const {
    // STUB — see T19.
    return *this;
}

} // namespace tensorforge
