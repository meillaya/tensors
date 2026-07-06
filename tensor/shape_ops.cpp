#include "tensor/shape_ops.hpp"

#include "tensor/Stride.hpp"

#include <numeric>
#include <stdexcept>

namespace tensorforge {

namespace {

bool is_contiguous(const Tensor& tensor) {
    const Shape& shape = tensor.shape();
    const Stride& stride = tensor.stride();
    if (shape.ndim() == 0) {
        return true;
    }
    const Stride expected = Stride::compute_row_major(shape);
    return stride == expected;
}

} // namespace

Tensor reshape(const Tensor& tensor, Shape new_shape) {
    const int64_t new_numel = new_shape.numel();
    if (new_numel != tensor.numel()) {
        throw std::invalid_argument("reshape: new shape must have same number of elements");
    }

    if (!is_contiguous(tensor)) {
        throw std::invalid_argument("reshape: only contiguous tensors can be reshaped as a view");
    }

    Stride new_stride = Stride::compute_row_major(new_shape);
    return Tensor(tensor.storage(), tensor.storage_offset(), std::move(new_shape), std::move(new_stride));
}

Tensor transpose(const Tensor& tensor, size_t dim0, size_t dim1) {
    const size_t ndim = tensor.shape().ndim();
    if (dim0 >= ndim || dim1 >= ndim) {
        throw std::out_of_range("transpose dimensions out of range");
    }

    Shape new_shape = tensor.shape();
    Stride new_stride = tensor.stride();
    std::swap(new_shape.data()[dim0], new_shape.data()[dim1]);
    std::swap(new_stride.data()[dim0], new_stride.data()[dim1]);

    return Tensor(tensor.storage(), tensor.storage_offset(), std::move(new_shape), std::move(new_stride));
}

Tensor permute(const Tensor& tensor, std::vector<size_t> dims) {
    const size_t ndim = tensor.shape().ndim();
    if (dims.size() != ndim) {
        throw std::invalid_argument("permute: dimension count must match tensor rank");
    }

    std::vector<bool> seen(ndim, false);
    for (size_t d : dims) {
        if (d >= ndim) {
            throw std::out_of_range("permute dimension out of range");
        }
        if (seen[d]) {
            throw std::invalid_argument("permute: dimensions must be a permutation");
        }
        seen[d] = true;
    }

    std::vector<int64_t> new_dims(ndim);
    std::vector<int64_t> new_strides(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        new_dims[i] = tensor.shape()[dims[i]];
        new_strides[i] = tensor.stride()[dims[i]];
    }

    return Tensor(tensor.storage(), tensor.storage_offset(), Shape(std::move(new_dims)), Stride(std::move(new_strides)));
}

} // namespace tensorforge
