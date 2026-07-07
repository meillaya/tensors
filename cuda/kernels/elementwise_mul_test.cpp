// TensorForge — elementwise mul test (Wave 4 / T18)

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

TEST_CASE("mul FP32: 2 * 3 = 6, 3 * 4 = 12") {
    Tensor a = arange(2, 5, 1, Dtype::Float32, Device::cuda(0));
    Tensor b = arange(3, 6, 1, Dtype::Float32, Device::cuda(0));
    Tensor c = Tensor::empty({3}, Dtype::Float32, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_mul(a.data(), b.data(), c.data(), 3, Dtype::Float32,
                             (void*)stream);
    cudaStreamSynchronize(stream);

    float out[3] = {0};
    cudaMemcpy(out, c.data(), sizeof(float) * 3, cudaMemcpyDeviceToHost);
    CHECK(out[0] == doctest::Approx(6.0f));
    CHECK(out[1] == doctest::Approx(12.0f));
    CHECK(out[2] == doctest::Approx(20.0f));
}

TEST_CASE("mul FP16: 0.5 * 2 = 1.0") {
    Tensor a = arange(0, 4, 1, Dtype::Float16, Device::cuda(0));
    Tensor b = arange(0, 4, 1, Dtype::Float16, Device::cuda(0));
    Tensor c = Tensor::empty({4}, Dtype::Float16, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_mul(a.data(), b.data(), c.data(), 4, Dtype::Float16,
                             (void*)stream);
    cudaStreamSynchronize(stream);

    __half out[4];
    cudaMemcpy(out, c.data(), sizeof(__half) * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; ++i) {
        CHECK(__half2float(out[i]) == doctest::Approx(1.0f * i * i).epsilon(0.05));
    }
}

TEST_CASE("mul BF16: 2 * 4 = 8") {
    Tensor a = arange(1, 5, 1, Dtype::BFloat16, Device::cuda(0));
    Tensor b = arange(2, 6, 1, Dtype::BFloat16, Device::cuda(0));
    Tensor c = Tensor::empty({4}, Dtype::BFloat16, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_mul(a.data(), b.data(), c.data(), 4, Dtype::BFloat16,
                             (void*)stream);
    cudaStreamSynchronize(stream);

    __nv_bfloat16 out[4];
    cudaMemcpy(out, c.data(), sizeof(__nv_bfloat16) * 4, cudaMemcpyDeviceToHost);
    CHECK(__bfloat162float(out[0]) == doctest::Approx(2.0f).epsilon(0.05));
    CHECK(__bfloat162float(out[1]) == doctest::Approx(6.0f).epsilon(0.05));
    CHECK(__bfloat162float(out[2]) == doctest::Approx(12.0f).epsilon(0.05));
    CHECK(__bfloat162float(out[3]) == doctest::Approx(20.0f).epsilon(0.05));
}

TEST_CASE("mul FP32: zero times anything is zero") {
    Tensor a = arange(0, 8, 1, Dtype::Float32, Device::cuda(0));
    Tensor b = zeros({8}, Dtype::Float32, Device::cuda(0));
    Tensor c = Tensor::empty({8}, Dtype::Float32, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    tensorforge::launch_mul(a.data(), b.data(), c.data(), 8, Dtype::Float32,
                             (void*)stream);
    cudaStreamSynchronize(stream);

    float out[8];
    cudaMemcpy(out, c.data(), sizeof(float) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) {
        CHECK(out[i] == doctest::Approx(0.0f));
    }
}

TEST_CASE("Tensor::operator+ on CUDA matches launch_add") {
    Tensor a = arange(1, 5, 1, Dtype::Float32, Device::cuda(0));
    Tensor b = arange(2, 6, 1, Dtype::Float32, Device::cuda(0));

    auto& ctx = DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);

    Tensor c = a + b;
    cudaStreamSynchronize(stream);

    float out[4];
    cudaMemcpy(out, c.data(), sizeof(float) * 4, cudaMemcpyDeviceToHost);
    CHECK(out[0] == doctest::Approx(3.0f));
    CHECK(out[1] == doctest::Approx(5.0f));
    CHECK(out[2] == doctest::Approx(7.0f));
    CHECK(out[3] == doctest::Approx(9.0f));
}