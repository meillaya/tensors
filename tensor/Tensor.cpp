// TensorForge — Tensor implementation (Wave 3 / T16-T28)
//
// Tensor.cpp is compiled by g++ (no CUDA toolchain). It uses the
// allocator_dispatch registry for CUDA-specific operations so it never needs
// to include cuda_runtime.h directly.

#include "tensor/Tensor.hpp"

#include "cuda/CudaKernelRegistry.hpp"
#include "tensor/AutogradWirer.hpp"
#include "tensor/CPUStorageAllocator.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tensorforge {

namespace {

void copy_cpu(const Tensor& src, Tensor& dst) {
    if (src.dtype() != dst.dtype()) {
        throw std::invalid_argument("Tensor::to cross-dtype not yet supported");
    }
    if (src.numel() != dst.numel()) {
        throw std::invalid_argument("Tensor::to shape mismatch");
    }
    std::memcpy(dst.data(), src.data(),
                static_cast<size_t>(src.numel()) * dtype_size(src.dtype()));
}

template <typename Op>
void elementwise_cpu(const Tensor& a, const Tensor& b, Tensor& out, Op op) {
    if (a.dtype() != out.dtype() || b.dtype() != out.dtype()) {
        throw std::invalid_argument("binary op dtype mismatch");
    }
    if (a.numel() != out.numel() || b.numel() != out.numel()) {
        throw std::invalid_argument("binary op shape mismatch");
    }
    int64_t n = a.numel();
    switch (a.dtype()) {
    case Dtype::Float32: {
        const float* ap = static_cast<const float*>(a.data());
        const float* bp = static_cast<const float*>(b.data());
        float* op_p = static_cast<float*>(out.data());
        for (int64_t i = 0; i < n; ++i) op_p[i] = op(ap[i], bp[i]);
        break;
    }
    default:
        throw std::invalid_argument("binary op dtype not yet implemented on CPU");
    }
}

struct AddOp {
    template <typename T>
    T operator()(T a, T b) const { return a + b; }
};

struct MulOp {
    template <typename T>
    T operator()(T a, T b) const { return a * b; }
};

// Module-private kernel registry storage (set by CUDA static initializer).
BinaryOpFn g_add_fn = nullptr;
BinaryOpFn g_mul_fn = nullptr;
UnaryOpFn g_relu_fn = nullptr;
UnaryOpFn g_sigmoid_fn = nullptr;
UnaryOpFn g_tanh_fn = nullptr;
LeakyReluFn g_leaky_relu_fn = nullptr;
void* g_current_stream = nullptr;

// Autograd wirer slots — see tensor/AutogradWirer.hpp for registration
// protocol.
AutogradWirerFn g_wire_add = nullptr;
AutogradWirerFn g_wire_mul = nullptr;

} // namespace

Tensor Tensor::empty(Shape shape, Dtype dtype, Device device) {
    const int64_t n = shape.numel();
    const size_t size_bytes = static_cast<size_t>(n) * dtype_size(dtype);

    Storage storage = allocator_dispatch::allocate(size_bytes, device, dtype);
    Stride stride = Stride::compute_row_major(shape);

    return Tensor(std::move(storage), 0, std::move(shape), std::move(stride));
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator+ shape mismatch");
    }
    Tensor out = Tensor::empty(shape_, dtype(), device());

    if (this->device().type == DeviceType::CPU) {
        elementwise_cpu(*this, other, out, AddOp{});
    } else {
        auto fn = g_add_fn;
        if (fn == nullptr) {
            throw std::runtime_error("CUDA add kernel not registered");
        }
        fn(this->data(), other.data(), out.data(), numel(), dtype(), g_current_stream);
    }

    if (g_wire_add != nullptr) {
        g_wire_add(out, *this, other);
    }

    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator* shape mismatch");
    }
    Tensor out = Tensor::empty(shape_, dtype(), device());

    if (this->device().type == DeviceType::CPU) {
        elementwise_cpu(*this, other, out, MulOp{});
    } else {
        auto fn = g_mul_fn;
        if (fn == nullptr) {
            throw std::runtime_error("CUDA mul kernel not registered");
        }
        fn(this->data(), other.data(), out.data(), numel(), dtype(), g_current_stream);
    }

    if (g_wire_mul != nullptr) {
        g_wire_mul(out, *this, other);
    }

    return out;
}

Tensor Tensor::to(Device device, Dtype dtype) const {
    if (device == this->device() && dtype == this->dtype()) {
        Tensor out = Tensor::empty(shape_, dtype, device);

        if (this->device().type == DeviceType::CPU) {
            copy_cpu(*this, out);
        } else {
            auto fn = allocator_dispatch::get_cuda_copy();
            if (fn == nullptr) {
                throw std::runtime_error("No CUDA copy callback registered");
            }
            size_t n_bytes = static_cast<size_t>(numel()) * dtype_size(dtype);
            fn(out.data(), this->data(), n_bytes,
               allocator_dispatch::CudaCopyKind::DeviceToDevice);
        }
        return out;
    }

    if (this->device().type == DeviceType::CPU && device.type == DeviceType::CPU) {
        Tensor out = Tensor::empty(shape_, dtype, device);
        copy_cpu(*this, out);
        return out;
    }

    if (this->dtype() != dtype) {
        throw std::invalid_argument("Tensor::to cross-dtype not yet implemented");
    }

    Tensor out = Tensor::empty(shape_, dtype, device);

    auto fn = allocator_dispatch::get_cuda_copy();
    if (fn == nullptr) {
        throw std::runtime_error("No CUDA copy callback registered; build did not link cuda allocator?");
    }

    allocator_dispatch::CudaCopyKind kind;
    if (this->device().type == DeviceType::CPU && device.type == DeviceType::CUDA) {
        kind = allocator_dispatch::CudaCopyKind::HostToDevice;
    } else if (this->device().type == DeviceType::CUDA && device.type == DeviceType::CPU) {
        kind = allocator_dispatch::CudaCopyKind::DeviceToHost;
    } else {
        kind = allocator_dispatch::CudaCopyKind::DeviceToDevice;
    }

    size_t n_bytes = static_cast<size_t>(numel()) * dtype_size(dtype);
    fn(out.data(), this->data(), n_bytes, kind);
    return out;
}

Tensor Tensor::to(Device device) const {
    return to(device, this->dtype());
}

Tensor Tensor::to(Dtype dtype) const {
    return to(this->device(), dtype);
}

Tensor Tensor::relu() const {
    Tensor out = Tensor::empty(shape_, dtype(), device());

    if (this->device().type == DeviceType::CPU) {
        const float* in_p = static_cast<const float*>(this->data());
        float* out_p = static_cast<float*>(out.data());
        for (int64_t i = 0; i < numel(); ++i) {
            out_p[i] = in_p[i] > 0.0f ? in_p[i] : 0.0f;
        }
        return out;
    }

    auto fn = g_relu_fn;
    if (fn == nullptr) {
        throw std::runtime_error("CUDA relu kernel not registered");
    }
    fn(this->data(), out.data(), numel(), dtype(), g_current_stream);
    return out;
}

Tensor Tensor::sigmoid() const {
    Tensor out = Tensor::empty(shape_, dtype(), device());

    if (this->device().type == DeviceType::CPU) {
        const float* in_p = static_cast<const float*>(this->data());
        float* out_p = static_cast<float*>(out.data());
        for (int64_t i = 0; i < numel(); ++i) {
            out_p[i] = 1.0f / (1.0f + std::exp(-in_p[i]));
        }
        return out;
    }

    auto fn = g_sigmoid_fn;
    if (fn == nullptr) {
        throw std::runtime_error("CUDA sigmoid kernel not registered");
    }
    fn(this->data(), out.data(), numel(), dtype(), g_current_stream);
    return out;
}

Tensor Tensor::tanh() const {
    Tensor out = Tensor::empty(shape_, dtype(), device());

    if (this->device().type == DeviceType::CPU) {
        const float* in_p = static_cast<const float*>(this->data());
        float* out_p = static_cast<float*>(out.data());
        for (int64_t i = 0; i < numel(); ++i) {
            out_p[i] = std::tanh(in_p[i]);
        }
        return out;
    }

    auto fn = g_tanh_fn;
    if (fn == nullptr) {
        throw std::runtime_error("CUDA tanh kernel not registered");
    }
    fn(this->data(), out.data(), numel(), dtype(), g_current_stream);
    return out;
}

Tensor Tensor::leaky_relu(float alpha) const {
    Tensor out = Tensor::empty(shape_, dtype(), device());

    if (this->device().type == DeviceType::CPU) {
        const float* in_p = static_cast<const float*>(this->data());
        float* out_p = static_cast<float*>(out.data());
        for (int64_t i = 0; i < numel(); ++i) {
            out_p[i] = in_p[i] > 0.0f ? in_p[i] : alpha * in_p[i];
        }
        return out;
    }

    auto fn = g_leaky_relu_fn;
    if (fn == nullptr) {
        throw std::runtime_error("CUDA leaky_relu kernel not registered");
    }
    fn(this->data(), out.data(), numel(), dtype(), alpha, g_current_stream);
    return out;
}

void register_cuda_add(BinaryOpFn fn) { g_add_fn = fn; }
void register_cuda_mul(BinaryOpFn fn) { g_mul_fn = fn; }
void register_cuda_relu(UnaryOpFn fn) { g_relu_fn = fn; }
void register_cuda_sigmoid(UnaryOpFn fn) { g_sigmoid_fn = fn; }
void register_cuda_tanh(UnaryOpFn fn) { g_tanh_fn = fn; }
void register_cuda_leaky_relu(LeakyReluFn fn) { g_leaky_relu_fn = fn; }

void register_current_stream(void* stream) { g_current_stream = stream; }

void register_add_wirer(AutogradWirerFn fn) { g_wire_add = fn; }
void register_mul_wirer(AutogradWirerFn fn) { g_wire_mul = fn; }

} // namespace tensorforge