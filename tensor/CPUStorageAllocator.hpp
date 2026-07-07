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

// ---------------------------------------------------------------------------
// Allocator dispatch (Wave 3 / T15)
// ---------------------------------------------------------------------------
//
// Tensor::empty calls into a registered allocator callback so the CPU code
// path doesn't need to include cuda_runtime.h. The CUDA allocator (or any
// other per-device allocator) registers itself at program startup. If no
// allocator is registered for a device, Tensor::empty throws.
//
// We use a function-pointer registration rather than a global allocator
// pointer so the CUDA library is loaded ONLY when an actual allocation
// request arrives (no DSO-load side-effects at startup time).
namespace allocator_dispatch {

using AllocateFn = Storage (*)(size_t size_bytes, Device device, Dtype dtype);
using FreeFn = void (*)(Storage storage);

// Register an allocator for a specific DeviceType. If multiple allocators
// are registered for the same DeviceType, the most recent wins. Registration
// is NOT thread-safe; do it from main() or a static initializer.
void register_allocator(DeviceType type, AllocateFn alloc_fn, FreeFn free_fn);

// Look up the registered allocator for `type`. Returns nullptr if none.
AllocateFn get_allocator(DeviceType type);
FreeFn get_deallocator(DeviceType type);

// Convenience: dispatch allocate based on the storage's device type.
[[nodiscard]] Storage allocate(size_t size_bytes, Device device, Dtype dtype);

// Dispatch free based on the storage's device type.
void free_storage(Storage storage);

// ---------------------------------------------------------------------------
// Cross-device async copy dispatch (Wave 3 / T16)
// ---------------------------------------------------------------------------
//
// Used by Tensor::to(device). Function-pointer registration keeps the CUDA
// runtime symbols out of tensor/ — the registered callback (provided by
// cuda/memory's static initializer) wraps cudaMemcpyAsync.

enum class CudaCopyKind : int {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
};

using CudaCopyFn = void (*)(void* dst, const void* src, size_t bytes,
                             CudaCopyKind kind);

void register_cuda_copy(CudaCopyFn fn);
CudaCopyFn get_cuda_copy();

// True if a CUDA copy callback has been registered.
bool cuda_copy_available();

} // namespace allocator_dispatch

} // namespace tensorforge
