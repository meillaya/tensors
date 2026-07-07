// TensorForge — CPU storage allocator (Wave 3 / T15)
//
// Aligned allocation for CPU storage + the allocator_dispatch registry that
// routes per-device allocations to the right backend. Tensor::empty calls
// into allocator_dispatch::allocate, which falls back to the CPU allocator
// when no CUDA allocator has been registered.

#include "tensor/CPUStorageAllocator.hpp"

#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>

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

struct AllocatorSlot {
    allocator_dispatch::AllocateFn allocate = nullptr;
    allocator_dispatch::FreeFn free = nullptr;
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

AllocatorSlot& slot_for(DeviceType type) {
    static AllocatorSlot slots[8] = {};
    return slots[static_cast<uint8_t>(type)];
}

Storage cpu_allocate_dispatch(size_t size_bytes, Device device, Dtype dtype) {
    return CPUStorageAllocator::instance().allocate(size_bytes, device, dtype);
}

void cpu_free_dispatch(Storage storage) {
    CPUStorageAllocator::instance().free(storage);
}

struct RegisterOnLoad {
    RegisterOnLoad() {
        allocator_dispatch::register_allocator(DeviceType::CPU,
                                               cpu_allocate_dispatch,
                                               cpu_free_dispatch);
    }
};
RegisterOnLoad g_register_cpu_on_load;

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

namespace allocator_dispatch {

void register_allocator(DeviceType type, AllocateFn alloc_fn, FreeFn free_fn) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto& slot = slot_for(type);
    slot.allocate = alloc_fn;
    slot.free = free_fn;
}

AllocateFn get_allocator(DeviceType type) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    return slot_for(type).allocate;
}

FreeFn get_deallocator(DeviceType type) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    return slot_for(type).free;
}

Storage allocate(size_t size_bytes, Device device, Dtype dtype) {
    AllocateFn fn = get_allocator(device.type);
    if (fn == nullptr) {
        throw std::runtime_error(
            "No allocator registered for device type. Did you link the CUDA "
            "allocator library? Or for CPU-only build, this is a programming error.");
    }
    return fn(size_bytes, device, dtype);
}

void free_storage(Storage storage) {
    if (!storage) {
        return;
    }
    FreeFn fn = get_deallocator(storage->device().type);
    if (fn == nullptr) {
        return;
    }
    fn(storage);
}

// ---------------------------------------------------------------------------
// CudaCopyFn registry
// ---------------------------------------------------------------------------

CudaCopyFn& cuda_copy_slot() {
    static CudaCopyFn fn = nullptr;
    return fn;
}

void register_cuda_copy(CudaCopyFn fn) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    cuda_copy_slot() = fn;
}

CudaCopyFn get_cuda_copy() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    return cuda_copy_slot();
}

bool cuda_copy_available() {
    return get_cuda_copy() != nullptr;
}

} // namespace allocator_dispatch

} // namespace tensorforge