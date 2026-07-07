// TensorForge — activation kernel tests (Wave 4 / T19)
//
// Smoke tests for launch_relu / launch_leaky_relu / launch_sigmoid /
// launch_tanh covering FP32 (precise) and FP16/BF16 (loose epsilon). All
// activations go through the unary kernel template with op functors
// defined in cuda/kernels/elementwise.cu.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda/CudaContext.hpp"
#include "cuda/kernels/elementwise.cuh"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using tensorforge::Device;
using tensorforge::DeviceContext;
using tensorforge::Dtype;
using tensorforge::Tensor;
using tensorforge::arange;
using tensorforge::full;
using tensorforge::ones;
using tensorforge::zeros;

namespace {

void sync_stream() {
    auto& ctx = tensorforge::DeviceContext::current();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.current_stream);
    cudaStreamSynchronize(stream);
}

}  // namespace

TEST_CASE("relu FP32: max(0, x) on signed range [-4..3]") {
    Tensor x = arange(-4, 4, 1, Dtype::Float32, Device::cuda(0));
    Tensor y = Tensor::empty({8}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_relu(x.data(), y.data(), 8, Dtype::Float32,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[8];
    cudaMemcpy(out, y.data(), sizeof(float) * 8, cudaMemcpyDeviceToHost);
    const float expected[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 8; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]));
    }
}

TEST_CASE("relu FP16: half-precision max(0, x)") {
    Tensor x = arange(-4, 4, 1, Dtype::Float16, Device::cuda(0));
    Tensor y = Tensor::empty({8}, Dtype::Float16, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_relu(x.data(), y.data(), 8, Dtype::Float16,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    __half out[8];
    cudaMemcpy(out, y.data(), sizeof(__half) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) {
        float expected = (i < 4) ? 0.0f : static_cast<float>(i - 4);
        CHECK(__half2float(out[i]) == doctest::Approx(expected).epsilon(0.05));
    }
}

TEST_CASE("relu BF16: brain-float max(0, x)") {
    Tensor x = arange(-2, 6, 1, Dtype::BFloat16, Device::cuda(0));
    Tensor y = Tensor::empty({8}, Dtype::BFloat16, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_relu(x.data(), y.data(), 8, Dtype::BFloat16,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    __nv_bfloat16 out[8];
    cudaMemcpy(out, y.data(), sizeof(__nv_bfloat16) * 8,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) {
        float expected = (i < 2) ? 0.0f : static_cast<float>(i - 2);
        CHECK(__bfloat162float(out[i]) == doctest::Approx(expected).epsilon(0.05));
    }
}

TEST_CASE("leaky_relu FP32 alpha=0.1: x if x>0 else 0.1*x") {
    Tensor x = arange(-4, 4, 1, Dtype::Float32, Device::cuda(0));
    Tensor y = Tensor::empty({8}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_leaky_relu(x.data(), y.data(), 8, Dtype::Float32, 0.1f,
                                   (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[8];
    cudaMemcpy(out, y.data(), sizeof(float) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) {
        float xi = static_cast<float>(i - 4);
        float expected = (xi > 0.0f) ? xi : 0.1f * xi;
        CHECK(out[i] == doctest::Approx(expected));
    }
}

TEST_CASE("sigmoid FP32: 1/(1+exp(-x)) at x=0 gives 0.5") {
    Tensor x = full({1}, 0.0f, Dtype::Float32, Device::cuda(0));
    Tensor y = Tensor::empty({1}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_sigmoid(x.data(), y.data(), 1, Dtype::Float32,
                                (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[1];
    cudaMemcpy(out, y.data(), sizeof(float) * 1, cudaMemcpyDeviceToHost);
    CHECK(out[0] == doctest::Approx(0.5f));
}

TEST_CASE("sigmoid FP32: known values match scipy-style reference") {
    // Hand-computed sigmoid on [-1.0, 0.0, 1.0, 2.0].
    Tensor x = full({4}, 0.0f, Dtype::Float32, Device::cuda(0));
    {
        float host_x[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
        auto& ctx = tensorforge::DeviceContext::current();
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
        cudaMemcpyAsync(x.data(), host_x, sizeof(float) * 4,
                        cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(ctx.current_stream));
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
    }

    Tensor y = Tensor::empty({4}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_sigmoid(x.data(), y.data(), 4, Dtype::Float32,
                                (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[4];
    cudaMemcpy(out, y.data(), sizeof(float) * 4, cudaMemcpyDeviceToHost);
    const float expected[4] = {
        1.0f / (1.0f + std::exp(1.0f)),
        0.5f,
        1.0f / (1.0f + std::exp(-1.0f)),
        1.0f / (1.0f + std::exp(-2.0f)),
    };
    for (int i = 0; i < 4; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-5));
    }
}

TEST_CASE("tanh FP32: known values match std::tanh") {
    Tensor x = full({5}, 0.0f, Dtype::Float32, Device::cuda(0));
    {
        float host_x[5] = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
        auto& ctx = tensorforge::DeviceContext::current();
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
        cudaMemcpyAsync(x.data(), host_x, sizeof(float) * 5,
                        cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(ctx.current_stream));
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));
    }

    Tensor y = Tensor::empty({5}, Dtype::Float32, Device::cuda(0));

    auto& ctx = tensorforge::DeviceContext::current();
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    tensorforge::launch_tanh(x.data(), y.data(), 5, Dtype::Float32,
                             (void*)ctx.current_stream);
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(ctx.current_stream));

    float out[5];
    cudaMemcpy(out, y.data(), sizeof(float) * 5, cudaMemcpyDeviceToHost);
    const float expected[5] = {
        std::tanh(-2.0f), std::tanh(-0.5f), 0.0f,
        std::tanh(0.5f), std::tanh(2.0f),
    };
    for (int i = 0; i < 5; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-5));
    }
}

TEST_CASE("Tensor::relu on CUDA matches launch_relu") {
    Tensor x = arange(-4, 4, 1, Dtype::Float32, Device::cuda(0));
    Tensor y = x.relu();
    sync_stream();

    float out[8];
    cudaMemcpy(out, y.data(), sizeof(float) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) {
        float xi = static_cast<float>(i - 4);
        CHECK(out[i] == doctest::Approx(xi > 0.0f ? xi : 0.0f));
    }
}

TEST_CASE("Tensor::sigmoid on CUDA at zero is 0.5") {
    Tensor x = full({1}, 0.0f, Dtype::Float32, Device::cuda(0));
    Tensor y = x.sigmoid();
    sync_stream();

    float out[1];
    cudaMemcpy(out, y.data(), sizeof(float) * 1, cudaMemcpyDeviceToHost);
    CHECK(out[0] == doctest::Approx(0.5f).epsilon(1e-5));
}

TEST_CASE("Tensor::leaky_relu on CUDA alpha=0.01") {
    Tensor x = arange(-4, 4, 1, Dtype::Float32, Device::cuda(0));
    Tensor y = x.leaky_relu(0.01f);
    sync_stream();

    float out[8];
    cudaMemcpy(out, y.data(), sizeof(float) * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) {
        float xi = static_cast<float>(i - 4);
        float expected = (xi > 0.0f) ? xi : 0.01f * xi;
        CHECK(out[i] == doctest::Approx(expected));
    }
}
