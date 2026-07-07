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
#include <vector>
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

struct SubOp {
    template <typename T>
    T operator()(T a, T b) const { return a - b; }
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
AutogradWirerFn g_wire_matmul = nullptr;
AutogradUnaryWirerFn g_wire_relu = nullptr;
AutogradUnaryWirerFn g_wire_sigmoid = nullptr;
AutogradUnaryWirerFn g_wire_tanh = nullptr;
AutogradUnaryWirerFn g_wire_leaky_relu = nullptr;
AutogradUnaryWirerFn g_wire_log = nullptr;
AutogradSoftmaxWirerFn g_wire_softmax = nullptr;
AutogradSoftmaxWirerFn g_wire_log_softmax = nullptr;
AutogradLayerNormWirerFn g_wire_layernorm = nullptr;

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

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator- shape mismatch");
    }
    Tensor out = Tensor::empty(shape_, dtype(), device());

    if (this->device().type == DeviceType::CPU) {
        elementwise_cpu(*this, other, out, SubOp{});
    } else {
        throw std::runtime_error("Tensor::operator- GPU not implemented");
    }
    // Subtraction is wired as a + (-1)*b in v1: the wirer for sub is
    // intentionally not registered, so users that need grad should use
    // add + negate or call matmul-style chains directly.
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


// ---------------------------------------------------------------------------
// T34 ops: matmul / softmax / log_softmax / layer_norm / log / sum / transpose
// ---------------------------------------------------------------------------

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.ndim() != 2 || other.shape_.ndim() != 2) {
        throw std::invalid_argument("Tensor::matmul: both operands must be 2D for v1");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Tensor::matmul: shape mismatch (a.cols != b.rows)");
    }
    if (this->device().type != DeviceType::CPU || other.device().type != DeviceType::CPU) {
        throw std::invalid_argument("Tensor::matmul: CPU only in v1");
    }
    if (dtype() != Dtype::Float32) {
        throw std::invalid_argument("Tensor::matmul: Float32 only in v1");
    }
    int64_t M = shape_[0];
    int64_t K = shape_[1];
    int64_t N = other.shape_[1];
    Tensor out = Tensor::empty(Shape{M, N}, dtype(), device());
    const float* a_p = static_cast<const float*>(this->data());
    const float* b_p = static_cast<const float*>(other.data());
    float* o_p = static_cast<float*>(out.data());
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += a_p[i * K + k] * b_p[k * N + j];
            }
            o_p[i * N + j] = acc;
        }
    }
    if (g_wire_matmul != nullptr) {
        g_wire_matmul(out, *this, other);
    }
    return out;
}

Tensor Tensor::softmax(int64_t dim) const {
    if (this->device().type != DeviceType::CPU || dtype() != Dtype::Float32) {
        throw std::invalid_argument("Tensor::softmax: CPU Float32 only in v1");
    }
    int64_t ndim = static_cast<int64_t>(shape_.ndim());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Tensor::softmax: dim out of range");
    }
    if (dim != ndim - 1) {
        throw std::invalid_argument("Tensor::softmax: only last-dim softmax in v1");
    }
    int64_t outer = numel() / shape_[ndim - 1];
    int64_t inner = shape_[ndim - 1];
    Tensor out = Tensor::empty(shape_, dtype(), device());
    const float* x_p = static_cast<const float*>(this->data());
    float* y_p = static_cast<float*>(out.data());
    for (int64_t r = 0; r < outer; ++r) {
        const float* row = x_p + r * inner;
        float* y_row = y_p + r * inner;
        float maxv = row[0];
        for (int64_t j = 1; j < inner; ++j) {
            if (row[j] > maxv) maxv = row[j];
        }
        float sumv = 0.0f;
        for (int64_t j = 0; j < inner; ++j) {
            y_row[j] = std::exp(row[j] - maxv);
            sumv += y_row[j];
        }
        float inv = 1.0f / sumv;
        for (int64_t j = 0; j < inner; ++j) {
            y_row[j] *= inv;
        }
    }
    if (g_wire_softmax != nullptr) {
        g_wire_softmax(out, *this, dim);
    }
    return out;
}

Tensor Tensor::log_softmax(int64_t dim) const {
    if (this->device().type != DeviceType::CPU || dtype() != Dtype::Float32) {
        throw std::invalid_argument("Tensor::log_softmax: CPU Float32 only in v1");
    }
    int64_t ndim = static_cast<int64_t>(shape_.ndim());
    if (dim < 0) dim += ndim;
    if (dim != ndim - 1) {
        throw std::invalid_argument("Tensor::log_softmax: only last-dim in v1");
    }
    int64_t outer = numel() / shape_[ndim - 1];
    int64_t inner = shape_[ndim - 1];
    Tensor out = Tensor::empty(shape_, dtype(), device());
    const float* x_p = static_cast<const float*>(this->data());
    float* y_p = static_cast<float*>(out.data());
    for (int64_t r = 0; r < outer; ++r) {
        const float* row = x_p + r * inner;
        float* y_row = y_p + r * inner;
        float maxv = row[0];
        for (int64_t j = 1; j < inner; ++j) {
            if (row[j] > maxv) maxv = row[j];
        }
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < inner; ++j) {
            float e = std::exp(row[j] - maxv);
            sum_exp += e;
        }
        float log_z = std::log(sum_exp) + maxv;
        for (int64_t j = 0; j < inner; ++j) {
            y_row[j] = row[j] - log_z;
        }
    }
    if (g_wire_log_softmax != nullptr) {
        g_wire_log_softmax(out, *this, dim);
    }
    return out;
}

Tensor Tensor::layer_norm(const Tensor& gamma, const Tensor& beta, float eps) const {
    if (this->device().type != DeviceType::CPU || dtype() != Dtype::Float32) {
        throw std::invalid_argument("Tensor::layer_norm: CPU Float32 only in v1");
    }
    if (shape_.ndim() != 2) {
        throw std::invalid_argument("Tensor::layer_norm: only 2D rows in v1");
    }
    int64_t rows = shape_[0];
    int64_t cols = shape_[1];
    if (gamma.numel() != cols || beta.numel() != cols) {
        throw std::invalid_argument("Tensor::layer_norm: gamma/beta must be [cols]");
    }
    Tensor out = Tensor::empty(shape_, dtype(), device());
    Tensor mean = Tensor::empty(Shape{rows}, dtype(), device());
    Tensor rstd = Tensor::empty(Shape{rows}, dtype(), device());
    const float* x_p = static_cast<const float*>(this->data());
    const float* g_p = static_cast<const float*>(gamma.data());
    const float* b_p = static_cast<const float*>(beta.data());
    float* y_p = static_cast<float*>(out.data());
    float* m_p = static_cast<float*>(mean.data());
    float* r_p = static_cast<float*>(rstd.data());
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = x_p + r * cols;
        float sum = 0.0f;
        for (int64_t j = 0; j < cols; ++j) sum += row[j];
        float m = sum / static_cast<float>(cols);
        float var_sum = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            float d = row[j] - m;
            var_sum += d * d;
        }
        float v = var_sum / static_cast<float>(cols);
        float rs = 1.0f / std::sqrt(v + eps);
        m_p[r] = m;
        r_p[r] = rs;
        float* y_row = y_p + r * cols;
        for (int64_t j = 0; j < cols; ++j) {
            y_row[j] = (row[j] - m) * rs * g_p[j] + b_p[j];
        }
    }
    if (g_wire_layernorm != nullptr) {
        g_wire_layernorm(out, *this, gamma, beta, eps);
    }
    return out;
}

Tensor Tensor::log() const {
    Tensor out = Tensor::empty(shape_, dtype(), device());
    if (this->device().type == DeviceType::CPU) {
        const float* x_p = static_cast<const float*>(this->data());
        float* o_p = static_cast<float*>(out.data());
        for (int64_t i = 0; i < numel(); ++i) {
            o_p[i] = std::log(x_p[i]);
        }
    } else {
        throw std::runtime_error("Tensor::log: GPU kernel not yet implemented");
    }
    if (g_wire_log != nullptr) {
        g_wire_log(out, *this);
    }
    return out;
}

Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    if (this->device().type != DeviceType::CPU || dtype() != Dtype::Float32) {
        throw std::invalid_argument("Tensor::sum: CPU Float32 only in v1");
    }
    int64_t ndim = static_cast<int64_t>(shape_.ndim());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Tensor::sum: dim out of range");
    }
    std::vector<int64_t> out_dims;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_dims.push_back(1);
        } else {
            out_dims.push_back(shape_[i]);
        }
    }
    Tensor out = Tensor::empty(Shape(out_dims), dtype(), device());
    const float* x_p = static_cast<const float*>(this->data());
    float* o_p = static_cast<float*>(out.data());

    std::vector<int64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape_[i + 1];
    }
    int64_t inner = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape_[i];
    int64_t outer = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape_[i];
    int64_t reduce = shape_[dim];

    int64_t out_numel = outer * inner;
    for (int64_t oi = 0; oi < out_numel; ++oi) {
        int64_t outer_idx = oi / inner;
        int64_t inner_idx = oi % inner;
        float s = 0.0f;
        for (int64_t d = 0; d < reduce; ++d) {
            int64_t idx = outer_idx * strides[dim] * reduce + d * strides[dim] + inner_idx;
            s += x_p[idx];
        }
        o_p[oi] = s;
    }
    return out;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    int64_t ndim = static_cast<int64_t>(shape_.ndim());
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::out_of_range("Tensor::transpose: dim out of range");
    }
    Shape new_shape = shape_;
    Stride new_stride = stride_;
    std::swap(new_shape.data()[dim0], new_shape.data()[dim1]);
    std::swap(new_stride.data()[dim0], new_stride.data()[dim1]);
    return Tensor(storage_, storage_offset_, std::move(new_shape), std::move(new_stride));
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
void register_matmul_wirer(AutogradWirerFn fn) { g_wire_matmul = fn; }
void register_relu_wirer(AutogradUnaryWirerFn fn) { g_wire_relu = fn; }
void register_sigmoid_wirer(AutogradUnaryWirerFn fn) { g_wire_sigmoid = fn; }
void register_tanh_wirer(AutogradUnaryWirerFn fn) { g_wire_tanh = fn; }
void register_leaky_relu_wirer(AutogradUnaryWirerFn fn) { g_wire_leaky_relu = fn; }
void register_log_wirer(AutogradUnaryWirerFn fn) { g_wire_log = fn; }
void register_softmax_wirer(AutogradSoftmaxWirerFn fn) { g_wire_softmax = fn; }
void register_log_softmax_wirer(AutogradSoftmaxWirerFn fn) { g_wire_log_softmax = fn; }
void register_layernorm_wirer(AutogradLayerNormWirerFn fn) { g_wire_layernorm = fn; }

} // namespace tensorforge