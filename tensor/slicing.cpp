#include "tensor/slicing.hpp"

#include <stdexcept>

namespace tensorforge {

namespace {

int64_t normalize_index(int64_t index, int64_t dim_size) {
    if (index < 0) {
        index += dim_size;
    }
    return index;
}

void check_dim(const Tensor& tensor, size_t dim) {
    if (dim >= static_cast<size_t>(tensor.shape().ndim())) {
        throw std::out_of_range("slice dimension out of range");
    }
}

} // namespace

Tensor slice(const Tensor& tensor, size_t dim, int64_t start, int64_t end) {
    check_dim(tensor, dim);

    const int64_t dim_size = tensor.shape()[dim];
    start = normalize_index(start, dim_size);
    end = normalize_index(end, dim_size);

    if (start < 0 || start > dim_size || end < 0 || end > dim_size || start > end) {
        throw std::out_of_range("slice range out of bounds");
    }

    Shape new_shape = tensor.shape();
    new_shape[dim] = end - start;

    Stride new_stride = tensor.stride();

    const size_t new_offset = tensor.storage_offset() + static_cast<size_t>(start) * static_cast<size_t>(tensor.stride()[dim]);

    return Tensor(tensor.storage(), new_offset, std::move(new_shape), std::move(new_stride));
}

Tensor select(const Tensor& tensor, size_t dim, int64_t index) {
    check_dim(tensor, dim);

    const int64_t dim_size = tensor.shape()[dim];
    index = normalize_index(index, dim_size);

    if (index < 0 || index >= dim_size) {
        throw std::out_of_range("select index out of bounds");
    }

    std::vector<int64_t> new_dims;
    std::vector<int64_t> new_strides;
    for (size_t i = 0; i < tensor.shape().ndim(); ++i) {
        if (i != dim) {
            new_dims.push_back(tensor.shape()[i]);
            new_strides.push_back(tensor.stride()[i]);
        }
    }

    const size_t new_offset = tensor.storage_offset() + static_cast<size_t>(index) * static_cast<size_t>(tensor.stride()[dim]);

    return Tensor(tensor.storage(), new_offset, Shape(std::move(new_dims)), Stride(std::move(new_strides)));
}

Tensor narrow(const Tensor& tensor, size_t dim, int64_t start, int64_t length) {
    if (length < 0) {
        throw std::out_of_range("narrow length must be non-negative");
    }
    return slice(tensor, dim, start, start + length);
}

} // namespace tensorforge
