// TensorForge — CudaStorageAllocator tests (Wave 3 / T15)
//
// Verifies cudaMallocAsync-backed allocation: addresses are non-null,
// different allocations don't alias, free is observable on next sync.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda/CudaContext.hpp"
#include "cuda/memory/CudaStorageAllocator.hpp"

using tensorforge::CudaStorageAllocator;
using tensorforge::Device;
using tensorforge::DeviceContext;
using tensorforge::Dtype;
using tensorforge::Storage;

TEST_CASE("CudaStorageAllocator allocates non-null device memory") {
    auto& alloc = CudaStorageAllocator::instance();
    Storage s = alloc.allocate(1024, Device::cuda(0), Dtype::Float32);
    REQUIRE(s != nullptr);
    CHECK(s->data() != nullptr);
    CHECK(s->size_bytes() == 1024);
    CHECK(s->device() == Device::cuda(0));
    CHECK(s->dtype() == Dtype::Float32);
}

TEST_CASE("CudaStorageAllocator distinct allocations don't alias") {
    auto& alloc = CudaStorageAllocator::instance();
    Storage a = alloc.allocate(256, Device::cuda(0), Dtype::Float32);
    Storage b = alloc.allocate(256, Device::cuda(0), Dtype::Float32);
    REQUIRE(a->data() != nullptr);
    REQUIRE(b->data() != nullptr);
    CHECK(a->data() != b->data());
}

TEST_CASE("CudaStorageAllocator zero-size allocation returns 1-byte storage") {
    auto& alloc = CudaStorageAllocator::instance();
    Storage s = alloc.allocate(0, Device::cuda(0), Dtype::Float32);
    REQUIRE(s != nullptr);
    CHECK(s->data() != nullptr);
    CHECK(s->size_bytes() == 1);
}

TEST_CASE("CudaStorageAllocator free + reallocate works") {
    auto& alloc = CudaStorageAllocator::instance();
    Storage s = alloc.allocate(4096, Device::cuda(0), Dtype::Float32);
    void* original_ptr = s->data();
    REQUIRE(original_ptr != nullptr);

    alloc.free(s);
    CHECK(s->data() == nullptr);

    Storage t = alloc.allocate(4096, Device::cuda(0), Dtype::Float32);
    REQUIRE(t != nullptr);
    CHECK(t->data() != nullptr);

    // Sync the stream so any pending cudaFreeAsync actually completes.
    auto& ctx = DeviceContext::current();
    cudaStreamSynchronize(ctx.current_stream);
}

TEST_CASE("CudaStorageAllocator rejects non-CUDA device") {
    auto& alloc = CudaStorageAllocator::instance();
    CHECK_THROWS_AS(alloc.allocate(64, Device::cpu(), Dtype::Float32), std::invalid_argument);
}

TEST_CASE("CudaStorageAllocator allocation is writable via device pointer") {
    auto& alloc = CudaStorageAllocator::instance();
    Storage s = alloc.allocate(sizeof(int32_t), Device::cuda(0), Dtype::Int32);
    REQUIRE(s->data() != nullptr);

    // Write a known value, read it back.
    int32_t host = 0x12345678;
    auto& ctx = DeviceContext::current();
    cudaMemcpyAsync(s->data(), &host, sizeof(int32_t), cudaMemcpyHostToDevice, ctx.current_stream);
    cudaStreamSynchronize(ctx.current_stream);

    int32_t read_back = 0;
    cudaMemcpyAsync(&read_back, s->data(), sizeof(int32_t), cudaMemcpyDeviceToHost, ctx.current_stream);
    cudaStreamSynchronize(ctx.current_stream);
    CHECK(read_back == host);
}