#pragma once

#include "tensor/Device.hpp"
#include "tensor/Dtype.hpp"
#include "tensor/Storage.hpp"

#include <cstddef>

namespace tensorforge {

class StorageAllocator {
public:
    virtual ~StorageAllocator() = default;

    [[nodiscard]] virtual Storage allocate(size_t size_bytes, Device device, Dtype dtype) = 0;

    virtual void free(Storage storage) = 0;
};

class CPUStorageAllocator : public StorageAllocator {
public:
    [[nodiscard]] Storage allocate(size_t size_bytes, Device device, Dtype dtype) override;

    void free(Storage storage) override;

    static CPUStorageAllocator& instance();
};

} // namespace tensorforge
