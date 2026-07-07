// TensorForge — elementwise add test (Wave 4 / T17)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContextForward.hpp"
#include "cuda/kernels/elementwise.cuh"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using tensorforge::Device;
using tensorforge::DeviceContext;
using tensorforge::Dtype;
using tensorforge::Tensor;
using tensorforge::arange;

TEST_CASE("add FP32 CPU staging: 1+2=3, 3+4=7") {
    Tensor a = arange(1, 4, 1, Dtype::Float32, Device::cuda(0));
    Tensor b = arange(2, 5, 1, Dtype::Float32, Device::cuda(0));
    Tensor c = Tensor::empty({3}, Dtype::Float32, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_add(a.data(), b.data(), c.data(), 3, Dtype::Float32,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[3] = {0};
    cudaMemcpy(out, c.data(), sizeof(float) * 3, cudaMemcpyDeviceToHost);
    CHECK(out[0] == doctest::Approx(3.0f));
    CHECK(out[1] == doctest::Approx(5.0f));
    CHECK(out[2] == doctest::Approx(7.0f));
}

TEST_CASE("add FP16: 0.5 + 0.25 = 0.75 (FP16 precision)") {
    Tensor a = arange(0, 8, 1, Dtype::Float16, Device::cuda(0));
    Tensor b = arange(0, 8, 1, Dtype::Float16, Device::cuda(0));
    Tensor c = Tensor::empty({8}, Dtype::Float16, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_add(a.data(), b.data(), c.data(), 8, Dtype::Float16,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    __half out[8];
    cudaMemcpy(out, c.data(), sizeof(__half) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) {
        CHECK(__half2float(out[i]) == doctest::Approx(2.0f * i).epsilon(0.05));
    }
}

TEST_CASE("add BF16: 2.0 + 3.0 = 5.0") {
    Tensor a = arange(0, 4, 1, Dtype::BFloat16, Device::cuda(0));
    Tensor b = arange(0, 4, 1, Dtype::BFloat16, Device::cuda(0));
    Tensor c = Tensor::empty({4}, Dtype::BFloat16, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_add(a.data(), b.data(), c.data(), 4, Dtype::BFloat16,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    __nv_bfloat16 out[4];
    cudaMemcpy(out, c.data(), sizeof(__nv_bfloat16) * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; ++i) {
        CHECK(__bfloat162float(out[i]) == doctest::Approx(2.0f * i).epsilon(0.05));
    }
}

TEST_CASE("add FP32: large tensor (1M elements)") {
    int64_t n = 1 << 20;
    Tensor a = arange(0, n, 1, Dtype::Float32, Device::cuda(0));
    Tensor b = arange(0, n, 1, Dtype::Float32, Device::cuda(0));
    Tensor c = Tensor::empty({n}, Dtype::Float32, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_add(a.data(), b.data(), c.data(), n, Dtype::Float32,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[3];
    cudaMemcpy(out, static_cast<char*>(c.data()) + (n - 3) * sizeof(float),
                sizeof(float) * 3, cudaMemcpyDeviceToHost);
    CHECK(out[0] == doctest::Approx(2.0f * (n - 3)));
    CHECK(out[1] == doctest::Approx(2.0f * (n - 2)));
    CHECK(out[2] == doctest::Approx(2.0f * (n - 1)));
}

TEST_CASE("add FP32: negative numbers work") {
    Tensor a = arange(-4, 0, 1, Dtype::Float32, Device::cuda(0));
    Tensor b = arange(0, 4, 1, Dtype::Float32, Device::cuda(0));
    Tensor c = Tensor::empty({4}, Dtype::Float32, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_add(a.data(), b.data(), c.data(), 4, Dtype::Float32,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[4];
    cudaMemcpy(out, c.data(), sizeof(float) * 4, cudaMemcpyDeviceToHost);
    CHECK(out[0] == doctest::Approx(-4.0f));
    CHECK(out[1] == doctest::Approx(-2.0f));
    CHECK(out[2] == doctest::Approx(0.0f));
    CHECK(out[3] == doctest::Approx(2.0f));
}