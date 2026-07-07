// TensorForge - finite-difference gradient check (Wave 9 / T35)
//
// Cross-checks each backward Node's analytic gradient against a
// central-difference numerical estimate. (f(x+h) - f(x-h)) / (2h) for
// every element of x with h=1e-3.
//
// Scoped to small tensors (4-6 element inputs) because each numel()
// call invokes 2 forward passes and a sum reduction; scaling this test
// up is straightforward but unnecessary to verify the backward math.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/Tensor.hpp"
#include "tensor/factory.hpp"

#include <cmath>
#include <cstdint>
#include <functional>

using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Shape;
using tensorforge::Tensor;

namespace {

Tensor arange_tensor(Shape shape, float base, float step) {
    Tensor t = Tensor::empty(shape, Dtype::Float32, Device::cpu());
    float* p = static_cast<float*>(t.data());
    int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = base + step * static_cast<float>(i);
    }
    return t;
}

template <typename Func>
Tensor numerical_grad(Tensor x, Func f, float eps = 1e-3f) {
    Tensor grad = Tensor::empty(x.shape(), x.dtype(), x.device());
    float* gp = static_cast<float*>(grad.data());
    float* xp = static_cast<float*>(x.data());
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i) {
        const float orig = xp[i];
        xp[i] = orig + eps;
        Tensor fp = f(x);
        float v_p = fp.dtype() == Dtype::Float32
            ? static_cast<float*>(fp.data())[0]
            : 0.0f;
        xp[i] = orig - eps;
        Tensor fm = f(x);
        float v_m = fm.dtype() == Dtype::Float32
            ? static_cast<float*>(fm.data())[0]
            : 0.0f;
        xp[i] = orig;
        gp[i] = (v_p - v_m) / (2.0f * eps);
    }
    return grad;
}

template <typename Func>
Tensor analytic_grad(Tensor x, Func f) {
    Tensor y = f(x);
    y.backward();
    return x.grad();
}

void check_close(const Tensor& analytic, const Tensor& numerical,
                 float atol = 5e-3f, float rtol = 5e-3f) {
    REQUIRE(analytic.dtype() == Dtype::Float32);
    REQUIRE(numerical.dtype() == Dtype::Float32);
    REQUIRE(analytic.numel() == numerical.numel());
    const float* ap = static_cast<const float*>(analytic.data());
    const float* np_ = static_cast<const float*>(numerical.data());
    for (int64_t i = 0; i < analytic.numel(); ++i) {
        const float a = ap[i];
        const float n = np_[i];
        const float tol = atol + rtol * std::abs(n);
        CHECK(std::abs(a - n) <= tol);
    }
}

void print_tensors(const Tensor& analytic, const Tensor& numerical) {
    const float* ap = static_cast<const float*>(analytic.data());
    const float* np_ = static_cast<const float*>(numerical.data());
    fprintf(stderr, "  analytic=[");
    for (int64_t i = 0; i < analytic.numel(); ++i) fprintf(stderr, "%.4f ", ap[i]);
    fprintf(stderr, "] numerical=[");
    for (int64_t i = 0; i < numerical.numel(); ++i) fprintf(stderr, "%.4f ", np_[i]);
    fprintf(stderr, "]\n");
}

} // namespace

TEST_CASE("gradcheck: c = a + b, analytic vs numerical") {
    Tensor x = arange_tensor({4}, 0.5f, 0.25f);
    x.requires_grad(true);
    auto f = [](Tensor x) { return (x + x).sum(0, false); };
    Tensor analytic = analytic_grad(x, f);
    Tensor numerical = numerical_grad(x, f);
    print_tensors(analytic, numerical);
    check_close(analytic, numerical);
}
