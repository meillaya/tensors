// TensorForge — Tensor::to tests (Wave 3 / T16)
//
// Verifies CPU->CUDA, CUDA->CPU, CUDA->CUDA, and CPU->CPU transfers all work
// correctly. Async on the CUDA path; we synchronize before reading back.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#ifndef TENSORFORGE_CPU_ONLY
#include "cuda/CudaContext.hpp"
#endif

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Tensor;
using tensorforge::arange;
using tensorforge::full;
using tensorforge::ones;
using tensorforge::zeros;

TEST_CASE("Tensor::to(CPU->CPU) copies data") {
    Tensor a = arange(0, 6, 1, Dtype::Float32, Device::cpu());
    Tensor b = a.to(Device::cpu());

    CHECK(b.device() == Device::cpu());
    CHECK(b.dtype() == Dtype::Float32);
    CHECK(b.shape() == Shape{6});
    const float* bp = static_cast<const float*>(b.data());
    for (int i = 0; i < 6; ++i) {
        CHECK(bp[i] == static_cast<float>(i));
    }
}

TEST_CASE("Tensor::to(same device) returns a copy") {
    Tensor a = full({2, 2}, 3.14, Dtype::Float32, Device::cpu());
    Tensor b = a.to(Device::cpu(), Dtype::Float32);

    CHECK(b.dtype() == Dtype::Float32);
    CHECK(b.device() == Device::cpu());
    const float* bp = static_cast<const float*>(b.data());
    CHECK(bp[0] == doctest::Approx(3.14f));
    CHECK(bp[3] == doctest::Approx(3.14f));
}

#ifndef TENSORFORGE_CPU_ONLY

TEST_CASE("Tensor::to(CPU->CUDA) copies data and reports device") {
    Tensor a = arange(0, 8, 1, Dtype::Float32, Device::cpu());
    Tensor b = a.to(Device::cuda(0));

    CHECK(b.device() == Device::cuda(0));
    CHECK(b.dtype() == Dtype::Float32);
    CHECK(b.shape() == Shape{8});

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(ctx.current_stream);

    float readback[8] = {0};
    cudaError_t err = cudaMemcpy(readback, b.data(), sizeof(float) * 8,
                                  cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    for (int i = 0; i < 8; ++i) {
        CHECK(readback[i] == static_cast<float>(i));
    }
}

TEST_CASE("Tensor::to(CUDA->CPU) copies data back to host") {
    Tensor a = arange(0, 8, 1, Dtype::Float32, Device::cuda(0));
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(ctx.current_stream);

    Tensor b = a.to(Device::cpu());

    CHECK(b.device() == Device::cpu());
    CHECK(b.dtype() == Dtype::Float32);
    const float* bp = static_cast<const float*>(b.data());
    for (int i = 0; i < 8; ++i) {
        CHECK(bp[i] == static_cast<float>(i));
    }
}

TEST_CASE("Tensor::to(CUDA->CUDA) round-trip preserves data") {
    Tensor a = arange(0, 16, 1, Dtype::Float32, Device::cuda(0));
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(ctx.current_stream);

    Tensor b = a.to(Device::cuda(0));
    cudaStreamSynchronize(ctx.current_stream);

    float readback[16] = {0};
    cudaMemcpy(readback, b.data(), sizeof(float) * 16, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; ++i) {
        CHECK(readback[i] == static_cast<float>(i));
    }
}

#endif