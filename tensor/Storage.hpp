#pragma once

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace tensorforge {

class StorageImpl {
public:
    StorageImpl(void* data, size_t size_bytes, Device device, Dtype dtype)
        : data_(data),
          size_bytes_(size_bytes),
          device_(device),
          dtype_(dtype),
          version_counter_(0) {}

    StorageImpl(const StorageImpl&) = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl(StorageImpl&&) = delete;
    StorageImpl& operator=(StorageImpl&&) = delete;

    ~StorageImpl() = default;

    [[nodiscard]] void* data() noexcept { return data_; }

    [[nodiscard]] const void* data() const noexcept { return data_; }

    [[nodiscard]] size_t size_bytes() const noexcept { return size_bytes_; }

    [[nodiscard]] Device device() const noexcept { return device_; }

    [[nodiscard]] Dtype dtype() const noexcept { return dtype_; }

    [[nodiscard]] uint32_t bump_version() noexcept {
        return version_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    [[nodiscard]] uint32_t current_version() const noexcept {
        return version_counter_.load(std::memory_order_relaxed);
    }

    void release() noexcept { data_ = nullptr; }

private:
    void* data_;
    size_t size_bytes_;
    Device device_;
    Dtype dtype_;
    std::atomic<uint32_t> version_counter_;
};

using Storage = std::shared_ptr<StorageImpl>;

} // namespace tensorforge
