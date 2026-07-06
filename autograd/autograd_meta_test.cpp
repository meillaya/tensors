#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "autograd/AccumulateGrad.hpp"
#include "autograd/AutogradMeta.hpp"
#include "tensor/Tensor.hpp"

#include <cstdint>
#include <memory>

using tensorforge::AccumulateGrad;
using tensorforge::AutogradMeta;
using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Tensor;

TEST_CASE("Tensor::requires_grad(true) creates AutogradMeta") {
    Tensor a = Tensor::empty({2, 3}, Dtype::Float32, Device::cpu());
    CHECK(!a.requires_grad());
    CHECK(a.autograd_meta() == nullptr);

    a.requires_grad(true);
    CHECK(a.requires_grad());
    CHECK(a.autograd_meta() != nullptr);
    CHECK(a.autograd_meta()->requires_grad_ == true);
}

TEST_CASE("requires_grad(false) does not create accumulator") {
    Tensor a = Tensor::empty({2, 3}, Dtype::Float32, Device::cpu());
    a.requires_grad(false);
    CHECK(!a.requires_grad());
    CHECK(a.autograd_meta() != nullptr);
    CHECK(a.autograd_meta()->grad_accumulator_ == nullptr);
}

TEST_CASE("grad_accumulator_ is wired correctly after requires_grad(true)") {
    Tensor a = Tensor::empty({2, 3}, Dtype::Float32, Device::cpu());
    a.requires_grad(true);

    const auto& acc = a.autograd_meta()->grad_accumulator_;
    CHECK(acc != nullptr);
    CHECK(acc->sequence_nr_ == UINT64_MAX);
}

TEST_CASE("grad_fn_ is null for leaf tensors") {
    Tensor a = Tensor::empty({2, 3}, Dtype::Float32, Device::cpu());
    a.requires_grad(true);
    CHECK(a.autograd_meta()->grad_fn_ == nullptr);
}

TEST_CASE("grad() returns empty tensor before backward") {
    Tensor a = Tensor::empty({2, 3}, Dtype::Float32, Device::cpu());
    a.requires_grad(true);
    CHECK(a.grad().numel() == 0);
}

TEST_CASE("AccumulateGrad writes gradient to variable") {
    Tensor a = Tensor::empty({2, 2}, Dtype::Float32, Device::cpu());
    a.requires_grad(true);

    const auto& acc = a.autograd_meta()->grad_accumulator_;
    REQUIRE(acc != nullptr);

    Tensor grad = Tensor::empty({2, 2}, Dtype::Float32, Device::cpu());
    float* ptr = static_cast<float*>(grad.data());
    for (int64_t i = 0; i < grad.numel(); ++i) {
        ptr[i] = static_cast<float>(i);
    }

    acc->apply(std::vector<Tensor>{std::move(grad)});

    Tensor result = a.grad();
    CHECK(result.numel() == 4);
    const float* rptr = static_cast<const float*>(result.data());
    CHECK(rptr[0] == 0.0f);
    CHECK(rptr[1] == 1.0f);
    CHECK(rptr[2] == 2.0f);
    CHECK(rptr[3] == 3.0f);
}
