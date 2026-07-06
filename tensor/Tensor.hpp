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

private:
    Storage storage_;
    size_t storage_offset_ = 0;
    Shape shape_;
    Stride stride_;
    std::shared_ptr<AutogradMeta> autograd_meta_;
};

} // namespace tensorforge
