// TensorForge — Tensor::empty on CUDA (Wave 3 / T15)
//
// Verifies that Tensor::empty dispatches to CudaStorageAllocator when given a
// CUDA device. This is the integration point of T15's allocator work.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdint>

#include <cuda_runtime.h>

#include "tensor/Tensor.hpp"

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Stride;
using tensorforge::Tensor;

TEST_CASE("Tensor::empty on CUDA(0) returns device-pointered tensor") {
    Tensor t = Tensor::empty({2, 3}, Dtype::Float32, Device::cuda(0));
    CHECK(t.numel() == 6);
    CHECK(t.shape() == Shape{2, 3});
    CHECK(t.dtype() == Dtype::Float32);
    CHECK(t.device() == Device::cuda(0));
    CHECK(t.data() != nullptr);
}

TEST_CASE("Tensor::empty on CUDA writes via async memcpy") {
    Tensor t = Tensor::empty({4}, Dtype::Int32, Device::cuda(0));
    REQUIRE(t.data() != nullptr);

    int32_t host_values[4] = {10, 20, 30, 40};
    cudaStream_t stream = 0;  // default stream
    cudaError_t err = cudaMemcpyAsync(t.data(), host_values, sizeof(int32_t) * 4,
                                       cudaMemcpyHostToDevice, stream);
    REQUIRE(err == cudaSuccess);
    err = cudaStreamSynchronize(stream);
    REQUIRE(err == cudaSuccess);

    int32_t readback[4] = {0, 0, 0, 0};
    err = cudaMemcpyAsync(readback, t.data(), sizeof(int32_t) * 4,
                          cudaMemcpyDeviceToHost, stream);
    REQUIRE(err == cudaSuccess);
    err = cudaStreamSynchronize(stream);
    REQUIRE(err == cudaSuccess);

    CHECK(readback[0] == 10);
    CHECK(readback[1] == 20);
    CHECK(readback[2] == 30);
    CHECK(readback[3] == 40);
}

TEST_CASE("Tensor::empty on CUDA(1) when only device 0 exists throws") {
    // We have exactly 1 H100. Asking for cuda(1) should throw.
    CHECK_THROWS(Tensor::empty({2}, Dtype::Float32, Device::cuda(1)));
}