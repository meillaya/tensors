#include "tensor/CPUStorageAllocator.hpp"

#include <cstdlib>
#include <stdexcept>

namespace tensorforge {

namespace {

constexpr size_t kAlignment = 64;

size_t round_up_to_multiple(size_t value, size_t multiple) {
    if (multiple == 0) {
        return value;
    }
    const size_t remainder = value % multiple;
    if (remainder == 0) {
        return value;
    }
    return value + (multiple - remainder);
}

} // namespace

Storage CPUStorageAllocator::allocate(size_t size_bytes, Device device, Dtype dtype) {
    if (device.type != DeviceType::CPU) {
        throw std::invalid_argument("CPUStorageAllocator can only allocate CPU storage");
    }

    const size_t aligned_size = round_up_to_multiple(size_bytes, kAlignment);
    void* ptr = std::aligned_alloc(kAlignment, aligned_size);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }

    return std::make_shared<StorageImpl>(ptr, size_bytes, device, dtype);
}

void CPUStorageAllocator::free(Storage storage) {
    if (!storage) {
        return;
    }
    void* ptr = storage->data();
    storage->release();
    std::free(ptr);
}

CPUStorageAllocator& CPUStorageAllocator::instance() {
    static CPUStorageAllocator instance;
    return instance;
}

} // namespace tensorforge
