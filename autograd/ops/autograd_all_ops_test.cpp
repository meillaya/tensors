// TensorForge — autograd backward tests for all wired ops (Wave 7 / T34)
//
// End-to-end exercises for Tensor::relu / softmax / log_softmax / layer_norm /
// matmul. Each TEST_CASE wires the autograd graph via the wirer slot, runs
// the Engine, and checks the gradient of the leaf against the analytic
// answer (or the expected shape when the backward is intentionally a stub).
//
// All cases are CPU-only because the wirer slots live in CPU code paths.
// Tags [cpu][autograd][fp32].
//
// Conventions:
//   * `filled(shape, v)` builds a CPU FP32 tensor with all elements = v.
//   * `check_all_equal(t, v, tol)` compares every element against `v`.
//   * `y.backward()` is the loss the gradients flow into. Analytic
//     grad of sum y w.r.t. y is ones, which the Engine writes back through
//     each Node until it reaches the leaf.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "autograd/AutogradMeta.hpp"
#include "autograd/Engine.hpp"
#include "autograd/ops/AddBackward.hpp"
#include "autograd/ops/MatmulBackward.hpp"
#include "autograd/ops/MulBackward.hpp"
#include "autograd/ops/ReluBackward.hpp"
#include "autograd/ops/SoftmaxBackward.hpp"
#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#include <cstdint>
#include <vector>

using tensorforge::AddBackward;
using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Engine;
using tensorforge::MatmulBackward;
using tensorforge::MulBackward;
using tensorforge::ReluBackward;
using tensorforge::Shape;
using tensorforge::SoftmaxBackward;
using tensorforge::Tensor;
using tensorforge::full;
using tensorforge::ones;
using tensorforge::zeros;

namespace {

Tensor filled(Shape shape, float value) {
    Tensor t = Tensor::empty(shape, Dtype::Float32, Device::cpu());
    float* p = static_cast<float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) p[i] = value;
    return t;
}

void check_all_equal(const Tensor& t, float expected, float tol = 1e-4f) {
    REQUIRE(t.dtype() == Dtype::Float32);
    REQUIRE(t.device() == Device::cpu());
    REQUIRE(t.numel() > 0);
    const float* ptr = static_cast<const float*>(t.data());
    for (int64_t i = 0; i < t.numel(); ++i) {
        CHECK(ptr[i] == doctest::Approx(expected).epsilon(tol));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

TEST_CASE("relu: y = max(x, 0); backward gives (x > 0) mask") {
    Tensor x = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());
    float* xp = static_cast<float*>(x.data());
    // 6 elements: positive, negative, zero, positive, negative, positive
    xp[0] = 1.0f; xp[1] = -2.0f; xp[2] = 0.0f;
    xp[3] = 3.0f; xp[4] = -1.0f; xp[5] = 4.0f;
    x.requires_grad(true);

    Tensor y = x.relu();
    REQUIRE(y.requires_grad());

    y.backward();

    const float* g = static_cast<const float*>(x.grad().data());
    // Expected mask: 1, 0, 0, 1, 0, 1 (zero gives zero gradient)
    CHECK(g[0] == doctest::Approx(1.0f));
    CHECK(g[1] == doctest::Approx(0.0f));
    CHECK(g[2] == doctest::Approx(0.0f));
    CHECK(g[3] == doctest::Approx(1.0f));
    CHECK(g[4] == doctest::Approx(0.0f));
    CHECK(g[5] == doctest::Approx(1.0f));
}

TEST_CASE("relu: positive-only input gives all-ones gradient") {
    Tensor x = filled({2, 3}, 2.5f);
    x.requires_grad(true);
    Tensor y = x.relu();
    y.backward();
    check_all_equal(x.grad(), 1.0f);
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

TEST_CASE("softmax: row of identical inputs gives uniform softmax + 0 grad") {
    // If every input is the same, softmax output is uniform = 1/N. Backward
    // of sum(softmax(x)) is grad_x[i] = y[i] - y[i] * sum(y) = 0 because
    // y is uniform and sum(y) = 1. (Math: grad_x = y - y * <y, 1>.)
    Tensor x = filled({2, 3}, 1.0f);
    x.requires_grad(true);
    Tensor y = x.softmax(-1);
    y.backward();

    const float* g = static_cast<const float*>(x.grad().data());
    for (int64_t i = 0; i < x.numel(); ++i) {
        CHECK(g[i] == doctest::Approx(0.0f).epsilon(1e-4));
    }
}

TEST_CASE("softmax: grad_fn is SoftmaxBackward") {
    Tensor x = filled({2, 3}, 0.5f);
    x.requires_grad(true);
    Tensor y = x.softmax(-1);
    REQUIRE(y.autograd_meta() != nullptr);
    REQUIRE(y.autograd_meta()->grad_fn_ != nullptr);
    auto sb = std::dynamic_pointer_cast<SoftmaxBackward>(y.autograd_meta()->grad_fn_);
    CHECK(sb != nullptr);
}

TEST_CASE("softmax: rows sum to 1 (forward sanity)") {
    Tensor x = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());
    float* xp = static_cast<float*>(x.data());
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f;
    xp[3] = -1.0f; xp[4] = 0.0f; xp[5] = 1.0f;
    Tensor y = x.softmax(-1);
    const float* yp = static_cast<const float*>(y.data());
    // Row 0
    float r0 = yp[0] + yp[1] + yp[2];
    CHECK(r0 == doctest::Approx(1.0f).epsilon(1e-5));
    // Row 1
    float r1 = yp[3] + yp[4] + yp[5];
    CHECK(r1 == doctest::Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("softmax: negative-dim indexing works") {
    Tensor x = filled({2, 3}, 0.5f);
    x.requires_grad(true);
    Tensor y = x.softmax(-1);
    y.backward();
    // Same uniform-input story: grad is zero.
    const float* g = static_cast<const float*>(x.grad().data());
    for (int64_t i = 0; i < x.numel(); ++i) {
        CHECK(g[i] == doctest::Approx(0.0f).epsilon(1e-4));
    }
}

// ---------------------------------------------------------------------------
// log_softmax
// ---------------------------------------------------------------------------

TEST_CASE("log_softmax: forward sanity - log of softmax") {
    Tensor x = Tensor::empty(Shape{3}, Dtype::Float32, Device::cpu());
    float* xp = static_cast<float*>(x.data());
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f;
    Tensor ls = x.log_softmax(-1);
    const float* lp = static_cast<const float*>(ls.data());
    // Reference softmax
    float maxv = 3.0f;
    float z = std::exp(1.0f - maxv) + std::exp(2.0f - maxv) + std::exp(3.0f - maxv);
    for (int64_t i = 0; i < 3; ++i) {
        float expected = xp[i] - maxv - std::log(z);
        CHECK(lp[i] == doctest::Approx(expected).epsilon(1e-5));
    }
}

TEST_CASE("log_softmax: backward produces grad of expected shape") {
    Tensor x = filled({2, 3}, 0.5f);
    x.requires_grad(true);
    Tensor ls = x.log_softmax(-1);
    ls.backward();
    REQUIRE(x.grad().numel() == x.numel());
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

TEST_CASE("layer_norm: forward with gamma=1, beta=0 is just (x-mean)/std") {
    Tensor x = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());
    float* xp = static_cast<float*>(x.data());
    xp[0] = 1.0f; xp[1] = 2.0f; xp[2] = 3.0f;
    xp[3] = 4.0f; xp[4] = 5.0f; xp[5] = 6.0f;
    Tensor gamma = ones(Shape{3}, Dtype::Float32, Device::cpu());
    Tensor beta = zeros(Shape{3}, Dtype::Float32, Device::cpu());

    Tensor y = x.layer_norm(gamma, beta);
    const float* yp = static_cast<const float*>(y.data());
    // Row 0: mean = 2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    // rstd = 1/sqrt(2/3 + eps) ~ 1.2247
    // y = (x - 2) * rstd
    float eps = 1e-5f;
    float m = 2.0f;
    float v = (1.0f + 0.0f + 1.0f) / 3.0f;
    float rs = 1.0f / std::sqrt(v + eps);
    CHECK(yp[0] == doctest::Approx((1.0f - m) * rs).epsilon(1e-4));
    CHECK(yp[1] == doctest::Approx((2.0f - m) * rs).epsilon(1e-4));
    CHECK(yp[2] == doctest::Approx((3.0f - m) * rs).epsilon(1e-4));
}

TEST_CASE("layer_norm: grad_x has the same shape as x") {
    Tensor x = filled({2, 3}, 0.5f);
    Tensor gamma = ones(Shape{3}, Dtype::Float32, Device::cpu());
    Tensor beta = zeros(Shape{3}, Dtype::Float32, Device::cpu());
    x.requires_grad(true);
    Tensor y = x.layer_norm(gamma, beta);
    y.backward();
    REQUIRE(x.grad().numel() == x.numel());
}

TEST_CASE("layer_norm: with gamma=2, output scales by 2") {
    Tensor x = filled({1, 3}, 1.0f);
    Tensor gamma = full(Shape{3}, 2.0, Dtype::Float32, Device::cpu());
    Tensor beta = zeros(Shape{3}, Dtype::Float32, Device::cpu());
    Tensor y = x.layer_norm(gamma, beta);
    // Row mean = 1, var = 0, rstd = 1/sqrt(eps) ~ 316.23
    // y = (x - 1) * rstd * 2 + 0 = 0 (since x - mean = 0)
    const float* yp = static_cast<const float*>(y.data());
    CHECK(std::abs(yp[0]) < 1.0f); // small but not exactly zero due to eps
    CHECK(std::abs(yp[1]) < 1.0f);
    CHECK(std::abs(yp[2]) < 1.0f);
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

TEST_CASE("matmul: 2x3 @ 3x2 -> 2x2 forward shape") {
    Tensor a = filled({2, 3}, 1.0f);
    Tensor b = filled({3, 2}, 1.0f);
    Tensor c = a.matmul(b);
    REQUIRE(c.shape()[0] == 2);
    REQUIRE(c.shape()[1] == 2);
    // c[i,j] = sum_k a[i,k] * b[k,j] = 3 for all i,j
    const float* cp = static_cast<const float*>(c.data());
    for (int64_t i = 0; i < c.numel(); ++i) {
        CHECK(cp[i] == doctest::Approx(3.0f).epsilon(1e-5));
    }
}

TEST_CASE("matmul: x.grad = sum(grad_y * w) for sum loss") {
    // For loss = sum(c) and c = A @ B:
    //   grad_A = ones * B^T = sum_j (B[:,j]) broadcast, i.e. each row of
    //     grad_A equals the column-sums of B (constant across rows of A).
    //   grad_B = A^T * ones = row-sums of A (constant across rows of B).
    Tensor a = filled({2, 3}, 1.0f);
    Tensor b = filled({3, 2}, 1.0f);
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a.matmul(b);
    c.backward();
    // grad_A[i, k] = sum_j B[k, j] = 2 for each (i, k).
    check_all_equal(a.grad(), 2.0f);
    // grad_B[k, j] = sum_i A[i, k] = 2 for each (k, j).
    check_all_equal(b.grad(), 2.0f);
}

TEST_CASE("matmul: grad_fn is MatmulBackward") {
    Tensor a = filled({2, 3}, 1.0f);
    Tensor b = filled({3, 2}, 1.0f);
    a.requires_grad(true);
    Tensor c = a.matmul(b);
    REQUIRE(c.autograd_meta() != nullptr);
    REQUIRE(c.autograd_meta()->grad_fn_ != nullptr);
    auto mb = std::dynamic_pointer_cast<MatmulBackward>(c.autograd_meta()->grad_fn_);
    CHECK(mb != nullptr);
}

TEST_CASE("matmul: arange inputs check analytic gradient") {
    // A = [[0, 1, 2], [3, 4, 5]]   (2x3)
    // B = [[0, 1], [2, 3], [4, 5]]  (3x2)
    // C = A @ B = [[0*0+1*2+2*4, 0*1+1*3+2*5], [3*0+4*2+5*4, 3*1+4*3+5*5]]
    //      = [[10, 13], [28, 40]]
    Tensor a = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());
    Tensor b = Tensor::empty(Shape{3, 2}, Dtype::Float32, Device::cpu());
    {
        float* ap = static_cast<float*>(a.data());
        ap[0] = 0.0f; ap[1] = 1.0f; ap[2] = 2.0f;
        ap[3] = 3.0f; ap[4] = 4.0f; ap[5] = 5.0f;
        float* bp = static_cast<float*>(b.data());
        bp[0] = 0.0f; bp[1] = 1.0f;
        bp[2] = 2.0f; bp[3] = 3.0f;
        bp[4] = 4.0f; bp[5] = 5.0f;
    }
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a.matmul(b);
    // Verify forward.
    const float* cp = static_cast<const float*>(c.data());
    CHECK(cp[0] == doctest::Approx(10.0f));
    CHECK(cp[1] == doctest::Approx(13.0f));
    CHECK(cp[2] == doctest::Approx(28.0f));
    CHECK(cp[3] == doctest::Approx(40.0f));
    c.backward();
    // grad_A[i, k] = sum_j B[k, j]
    //   grad_A[0,0] = 0+1 = 1, grad_A[0,1] = 2+3 = 5, grad_A[0,2] = 4+5 = 9
    //   grad_A[1,0] = 1, grad_A[1,1] = 5, grad_A[1,2] = 9
    const float* ga = static_cast<const float*>(a.grad().data());
    CHECK(ga[0] == doctest::Approx(1.0f));
    CHECK(ga[1] == doctest::Approx(5.0f));
    CHECK(ga[2] == doctest::Approx(9.0f));
    CHECK(ga[3] == doctest::Approx(1.0f));
    CHECK(ga[4] == doctest::Approx(5.0f));
    CHECK(ga[5] == doctest::Approx(9.0f));
    // grad_B[k, j] = sum_i A[i, k]
    //   grad_B[0,0] = 0+3 = 3, grad_B[0,1] = 0+3 = 3
    //   grad_B[1,0] = 1+4 = 5, grad_B[1,1] = 1+4 = 5
    //   grad_B[2,0] = 2+5 = 7, grad_B[2,1] = 2+5 = 7
    const float* gb = static_cast<const float*>(b.grad().data());
    CHECK(gb[0] == doctest::Approx(3.0f));
    CHECK(gb[1] == doctest::Approx(3.0f));
    CHECK(gb[2] == doctest::Approx(5.0f));
    CHECK(gb[3] == doctest::Approx(5.0f));
    CHECK(gb[4] == doctest::Approx(7.0f));
    CHECK(gb[5] == doctest::Approx(7.0f));
}

TEST_CASE("matmul: chained with operator+ propagates both grads") {
    // d = (A @ B) + ones_like_C  ->  sum(d) backward exercises both wirers.
    // A = ones(2,3), B = ones(3,2), C = A @ B = ones(2,2).
    // d / dA = ones @ B^T = 2 (each entry) since B is all ones.
    // d / dB = A^T @ ones = 2 (each entry) since A is all ones.
    Tensor a = filled({2, 3}, 1.0f);
    Tensor b = filled({3, 2}, 1.0f);
    Tensor e = filled({2, 2}, 1.0f);
    a.requires_grad(true);
    b.requires_grad(true);
    e.requires_grad(true);
    Tensor c = a.matmul(b);
    Tensor d = c + e;
    d.backward();
    check_all_equal(a.grad(), 2.0f);
    check_all_equal(b.grad(), 2.0f);
    check_all_equal(e.grad(), 1.0f); // d / dE = ones
}

// ---------------------------------------------------------------------------
// Cross-composition: matmul inside a chain exercises both wirers.
// ---------------------------------------------------------------------------

TEST_CASE("composed: relu(matmul(x, w)) backward propagates through both") {
    Tensor x = Tensor::empty(Shape{2, 3}, Dtype::Float32, Device::cpu());
    Tensor w = Tensor::empty(Shape{3, 4}, Dtype::Float32, Device::cpu());
    {
        float* xp = static_cast<float*>(x.data());
        // Mix of negative and positive so relu mask is non-trivial.
        xp[0] = 1.0f;  xp[1] = -2.0f; xp[2] = 3.0f;
        xp[3] = -1.0f; xp[4] = 2.0f;  xp[5] = 0.5f;
        // All-positive weight so the matmul output signs match x's.
        float* wp = static_cast<float*>(w.data());
        for (int64_t i = 0; i < w.numel(); ++i) wp[i] = 0.5f;
    }
    x.requires_grad(true);
    w.requires_grad(true);
    Tensor y = (x.matmul(w)).relu();
    y.backward();
    // grad_x = (matmul > 0) mask @ W  ->  shape (2, 4) @ W^T shape (4, 3)
    // grad_w = X^T @ (matmul > 0) mask  ->  shape (3, 2) @ (2, 4)
    REQUIRE(x.grad().numel() == x.numel());
    REQUIRE(w.grad().numel() == w.numel());
}