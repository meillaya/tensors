// TensorForge — CudaStorageAllocator implementation (Wave 3 / T15)

#include "cuda/memory/CudaStorageAllocator.hpp"

#include "cuda/CudaContext.hpp"
#include "tensor/CPUStorageAllocator.hpp"

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include <cuda_runtime.h>

namespace tensorforge {

namespace {

struct DevicePool {
    cudaMemPool_t pool = nullptr;
    bool initialized = false;
};

std::unordered_map<int32_t, DevicePool>& pool_map() {
    static std::unordered_map<int32_t, DevicePool> map;
    return map;
}

std::mutex& pool_mutex() {
    static std::mutex m;
    return m;
}

void ensure_pool_for_device(int32_t device_index) {
    std::lock_guard<std::mutex> lock(pool_mutex());
    auto& map = pool_map();
    auto& slot = map[device_index];
    if (slot.initialized) {
        return;
    }

    cudaError_t err = cudaDeviceGetDefaultMemPool(&slot.pool, device_index);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceGetDefaultMemPool failed: ") +
                                  cudaGetErrorString(err));
    }

    uint64_t release_threshold = 64ull * 1024 * 1024;
    err = cudaMemPoolSetAttribute(slot.pool, cudaMemPoolAttrReleaseThreshold,
                                   &release_threshold);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemPoolSetAttribute failed: ") +
                                  cudaGetErrorString(err));
    }

    slot.initialized = true;
}

Storage cuda_allocate_dispatch(size_t size_bytes, Device device, Dtype dtype) {
    return CudaStorageAllocator::instance().allocate(size_bytes, device, dtype);
}

void cuda_free_dispatch(Storage storage) {
    CudaStorageAllocator::instance().free(storage);
}

void cuda_copy_dispatch(void* dst, const void* src, size_t bytes,
                         allocator_dispatch::CudaCopyKind kind) {
    auto& ctx = DeviceContext::current();
    cudaStream_t stream = ctx.current_stream;
    cudaMemcpyKind cuda_kind = cudaMemcpyHostToHost;
    switch (kind) {
    case allocator_dispatch::CudaCopyKind::HostToHost:
        cuda_kind = cudaMemcpyHostToHost; break;
    case allocator_dispatch::CudaCopyKind::HostToDevice:
        cuda_kind = cudaMemcpyHostToDevice; break;
    case allocator_dispatch::CudaCopyKind::DeviceToHost:
        cuda_kind = cudaMemcpyDeviceToHost; break;
    case allocator_dispatch::CudaCopyKind::DeviceToDevice:
        cuda_kind = cudaMemcpyDeviceToDevice; break;
    }
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cuda_kind, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                                  cudaGetErrorString(err));
    }
}

struct RegisterOnLoad {
    RegisterOnLoad() {
        allocator_dispatch::register_allocator(DeviceType::CUDA,
                                               cuda_allocate_dispatch,
                                               cuda_free_dispatch);
        allocator_dispatch::register_cuda_copy(cuda_copy_dispatch);
    }
};
RegisterOnLoad g_register_cuda_on_load;

} // namespace

Storage CudaStorageAllocator::allocate(size_t size_bytes, Device device, Dtype dtype) {
    if (device.type != DeviceType::CUDA) {
        throw std::invalid_argument("CudaStorageAllocator can only allocate CUDA storage");
    }
    if (size_bytes == 0) {
        size_bytes = 1;
    }

    ensure_pool_for_device(device.index);

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = ctx.current_stream;

    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, size_bytes, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocAsync failed: ") +
                                  cudaGetErrorString(err));
    }
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }

    return std::make_shared<StorageImpl>(ptr, size_bytes, device, dtype);
}

void CudaStorageAllocator::free(Storage storage) {
    if (!storage) {
        return;
    }
    Device device = storage->device();
    if (device.type != DeviceType::CUDA) {
        throw std::invalid_argument("CudaStorageAllocator::free on non-CUDA storage");
    }
    void* ptr = storage->data();
    storage->release();
    if (ptr != nullptr) {
        auto& ctx = DeviceContext::current();
        cudaStream_t stream = ctx.current_stream;
        cudaError_t err = cudaFreeAsync(ptr, stream);
        (void)err;
    }
}

CudaStorageAllocator& CudaStorageAllocator::instance() {
    static CudaStorageAllocator instance;
    return instance;
}

} // namespace tensorforge