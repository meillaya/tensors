// TensorForge — CudaStorageAllocator (Wave 3 / T15)
//
// GPU storage allocator backed by cudaMallocAsync + per-device cudaMemPool.
// Pools reuse memory within a device (no per-call cudaMalloc/cudaFree
// handshake with the driver), and the release-threshold keeps the pool from
// hoarding memory indefinitely.
//
// Each (device, stream) pair gets its allocation routed through the device's
// pool; cudaMallocAsync/FreeAsync are stream-aware so we can free before the
// kernel that wrote to the buffer is done.

#pragma once

#include "tensor/CPUStorageAllocator.hpp"

#include <cstddef>

namespace tensorforge {

class CudaStorageAllocator : public StorageAllocator {
public:
    [[nodiscard]] Storage allocate(size_t size_bytes, Device device, Dtype dtype) override;
    void free(Storage storage) override;

    static CudaStorageAllocator& instance();

private:
    CudaStorageAllocator() = default;
};

} // namespace tensorforge