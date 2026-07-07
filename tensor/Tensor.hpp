#pragma once

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Shape.hpp"
#include "tensor/Storage.hpp"
#include "tensor/Stride.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tensorforge {

struct AutogradMeta;

class Tensor {
public:
    Tensor() = default;

    Tensor(Storage storage, size_t storage_offset, Shape shape, Stride stride)
        : storage_(std::move(storage)),
          storage_offset_(storage_offset),
          shape_(std::move(shape)),
          stride_(std::move(stride)) {}

    [[nodiscard]] static Tensor empty(Shape shape, Dtype dtype, Device device);

    [[nodiscard]] void* data() noexcept { return storage_ ? static_cast<char*>(storage_->data()) + storage_offset_ * dtype_size(dtype()) : nullptr; }

    [[nodiscard]] const void* data() const noexcept {
        return storage_ ? static_cast<const char*>(storage_->data()) + storage_offset_ * dtype_size(dtype()) : nullptr;
    }

    [[nodiscard]] int64_t numel() const noexcept { return shape_.numel(); }

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }

    [[nodiscard]] Shape& shape() noexcept { return shape_; }

    [[nodiscard]] const Stride& stride() const noexcept { return stride_; }

    [[nodiscard]] Stride& stride() noexcept { return stride_; }

    [[nodiscard]] Dtype dtype() const noexcept { return storage_ ? storage_->dtype() : Dtype::Float32; }

    [[nodiscard]] Device device() const noexcept { return storage_ ? storage_->device() : Device::cpu(); }

    [[nodiscard]] uint32_t version() const noexcept { return storage_ ? storage_->current_version() : 0; }

    [[nodiscard]] size_t storage_offset() const noexcept { return storage_offset_; }

    [[nodiscard]] Storage storage() const noexcept { return storage_; }

    [[nodiscard]] uint32_t bump_version() noexcept {
        if (storage_) {
            return storage_->bump_version();
        }
        return 0;
    }

    [[nodiscard]] std::shared_ptr<AutogradMeta>& autograd_meta() noexcept;
    [[nodiscard]] const std::shared_ptr<AutogradMeta>& autograd_meta() const noexcept;

    void requires_grad(bool req);
    [[nodiscard]] bool requires_grad() const noexcept;
    [[nodiscard]] Tensor grad() const;
    void backward();

    // Elementwise activations. Forward returns the activated tensor;
    // the wirer slot installs the matching backward Node (ReluBackward,
    // etc.) when any input requires_grad.
    [[nodiscard]] Tensor relu() const;
    [[nodiscard]] Tensor sigmoid() const;
    [[nodiscard]] Tensor tanh() const;
    [[nodiscard]] Tensor leaky_relu(float alpha) const;

    // Move/clone across devices or dtypes. Async on the current device's
    // stream — call cudaStreamSynchronize (or equivalent) before reading the
    // result on the host.
    [[nodiscard]] Tensor to(Device device) const;
    [[nodiscard]] Tensor to(Dtype dtype) const;
    [[nodiscard]] Tensor to(Device device, Dtype dtype) const;

    // Elementwise binary ops (T17/T18). Both operands must share shape and
    // dtype and device. Output is a new tensor. The wirer slot installs
    // AddBackward / MulBackward when requires_grad is true.
    [[nodiscard]] Tensor operator+(const Tensor& other) const;
    [[nodiscard]] Tensor operator-(const Tensor& other) const;
    [[nodiscard]] Tensor operator*(const Tensor& other) const;

    // T34: 2D matrix multiply. Both operands must be 2D Float32 on CPU.
    // Wirer installs MatmulBackward when requires_grad is true.
    [[nodiscard]] Tensor matmul(const Tensor& other) const;

    // T34: softmax along `dim` (negative indexing supported). CPU Float32
    // for v1. Wirer installs SoftmaxBackward when requires_grad is true.
    [[nodiscard]] Tensor softmax(int64_t dim) const;

    // T34: log_softmax along `dim` (negative indexing supported).
    [[nodiscard]] Tensor log_softmax(int64_t dim) const;

    // T34: row-wise layer-norm. `gamma` and `beta` are 1D [last_dim]
    // vectors on the same device.
    [[nodiscard]] Tensor layer_norm(const Tensor& gamma, const Tensor& beta,
                                     float eps = 1e-5f) const;

    // T34: elementwise natural log. CPU Float32 for v1.
    [[nodiscard]] Tensor log() const;

    // T34: sum reduction along `dim` with optional keepdim. Negative
    // indexing supported. CPU Float32 for v1.
    [[nodiscard]] Tensor sum(int64_t dim, bool keepdim = false) const;

    // T46: scalar mean over all elements. CPU Float32 for v1.
    [[nodiscard]] Tensor mean() const;

    // T34: instance transpose with negative-index support. Returns a view
    // over the same storage with swapped shape/stride.
    [[nodiscard]] Tensor transpose(int64_t dim0, int64_t dim1) const;

private:
    Storage storage_;
    size_t storage_offset_ = 0;
    Shape shape_;
    Stride stride_;
    std::shared_ptr<AutogradMeta> autograd_meta_;
};

} // namespace tensorforge
