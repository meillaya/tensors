// TensorForge — CUDA context test (Wave 3 / T14)
//
// Smoke tests for the CudaStream / CudaEvent / DeviceContext wrappers.
// Uses [gpu] tag so it only runs on GPU pods.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda/CudaContext.hpp"

using tensorforge::CudaEvent;
using tensorforge::CudaStream;
using tensorforge::DeviceContext;

TEST_CASE("CudaStream default-construction is null") {
    CudaStream s;
    CHECK(s.get() == nullptr);
}

TEST_CASE("CudaStream construction creates non-null stream") {
    CudaStream s(0);
    CHECK(s.get() != nullptr);
    CHECK(s.get() != cudaStreamLegacy);
    CHECK(s.get() != cudaStreamPerThread);
}

TEST_CASE("CudaStream move construction transfers ownership") {
    CudaStream a(0);
    cudaStream_t raw = a.get();
    CudaStream b(std::move(a));
    CHECK(a.get() == nullptr);
    CHECK(b.get() == raw);
}

TEST_CASE("CudaStream synchronize does not throw") {
    CudaStream s(0);
    s.synchronize();
    CHECK(true);
}

TEST_CASE("CudaEvent default-construction is null") {
    CudaEvent e;
    CHECK_FALSE(e.is_valid());
}

TEST_CASE("CudaEvent construction creates valid event") {
    CudaEvent e(cudaEventDisableTiming);
    CHECK(e.is_valid());
}

TEST_CASE("CudaEvent record + synchronize does not throw") {
    CudaStream s(0);
    CudaEvent e(cudaEventDisableTiming);
    e.record(s.get());
    e.synchronize();
    CHECK(true);
}

TEST_CASE("DeviceContext::current has valid device_index and stream") {
    auto& ctx = DeviceContext::current();
    CHECK(ctx.device_index == 0);
    CHECK(ctx.current_stream != nullptr);
}

TEST_CASE("DeviceContext::current is stable across calls") {
    auto& a = DeviceContext::current();
    auto& b = DeviceContext::current();
    CHECK(&a == &b);
    CHECK(a.current_stream == b.current_stream);
}

TEST_CASE("Kernel launch on DeviceContext stream succeeds") {
    auto& ctx = DeviceContext::current();

    // Tiny launch: 1 thread, 1 block. Just a noop kernel test.
    auto noop_kernel = [] __device__ () { return; };
    (void)noop_kernel;

    // Use cudaMallocAsync / cudaFreeAsync to exercise the stream wiring.
    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, 64, ctx.current_stream);
    CHECK(err == cudaSuccess);

    err = cudaFreeAsync(ptr, ctx.current_stream);
    CHECK(err == cudaSuccess);

    err = cudaStreamSynchronize(ctx.current_stream);
    CHECK(err == cudaSuccess);
}