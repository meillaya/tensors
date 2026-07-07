#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "nn/CrossEntropyLoss.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

using namespace tensorforge;

TEST_CASE("CrossEntropyLoss forward — single sample") {
    nn::CrossEntropyLoss cel;
    // logits = [1.0, 2.0, 3.0]  →  softmax ≈ [0.0900, 0.2447, 0.6652]
    // target = 2  →  loss = -log(0.6652) ≈ 0.4076
    Tensor logits = full({1, 3}, 1.0f, Dtype::Float32, Device::cpu());
    float* lp = static_cast<float*>(logits.data());
    lp[1] = 2.0f;
    lp[2] = 3.0f;

    Tensor targets = Tensor::empty({1}, Dtype::Int64, Device::cpu());
    static_cast<int64_t*>(targets.data())[0] = 2;

    Tensor loss = cel.forward(logits, targets);
    REQUIRE(loss.shape().numel() == 1);
    const float loss_val = static_cast<const float*>(loss.data())[0];
    MESSAGE("computed loss = ", loss_val);
    CHECK(loss_val == doctest::Approx(0.4076f).epsilon(0.01f));
}

TEST_CASE("CrossEntropyLoss forward — batch") {
    nn::CrossEntropyLoss cel;
    // Two samples with uniform logits [1, 1, 1]  →  softmax = [1/3, 1/3, 1/3]
    // target[0] = 0  →  -log(1/3) ≈ 1.0986
    // target[1] = 2  →  -log(1/3) ≈ 1.0986
    // mean ≈ 1.0986
    Tensor logits = full({2, 3}, 1.0f, Dtype::Float32, Device::cpu());

    Tensor targets = Tensor::empty({2}, Dtype::Int64, Device::cpu());
    int64_t* tp = static_cast<int64_t*>(targets.data());
    tp[0] = 0;
    tp[1] = 2;

    Tensor loss = cel.forward(logits, targets);
    REQUIRE(loss.shape().numel() == 1);
    const float loss_val = static_cast<const float*>(loss.data())[0];
    MESSAGE("batch mean loss = ", loss_val);
    // Just check it's finite and positive.
    CHECK(loss_val > 0.0f);
    CHECK(loss_val == doctest::Approx(1.0986f).epsilon(0.01f));
}

TEST_CASE("CrossEntropyLoss 1-arg form throws") {
    nn::CrossEntropyLoss cel;
    Tensor logits = full({1, 3}, 1.0f, Dtype::Float32, Device::cpu());
    CHECK_THROWS(static_cast<void>(cel.forward(logits)));
}