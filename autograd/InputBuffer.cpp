#include "autograd/InputBuffer.hpp"

#include "tensor/Dtype.hpp"

#include <stdexcept>

namespace tensorforge {

namespace {

Tensor add_tensors(const Tensor& a, const Tensor& b) {
    if (a.dtype() != Dtype::Float32 || b.dtype() != Dtype::Float32) {
        throw std::runtime_error("InputBuffer: gradient accumulation only supports Float32 in v1");
    }
    if (a.shape() != b.shape()) {
        throw std::runtime_error("InputBuffer: shape mismatch during gradient accumulation");
    }
    Tensor result = Tensor::empty(a.shape(), a.dtype(), a.device());
    const float* pa = static_cast<const float*>(a.data());
    const float* pb = static_cast<const float*>(b.data());
    float* pr = static_cast<float*>(result.data());
    for (int64_t i = 0; i < a.numel(); ++i) {
        pr[i] = pa[i] + pb[i];
    }
    return result;
}

} // namespace

InputBuffer::InputBuffer(size_t num_outputs) : buffer_(num_outputs) {}

void InputBuffer::add(size_t pos, Tensor grad) {
    if (pos >= buffer_.size()) {
        buffer_.resize(pos + 1);
    }
    if (!buffer_[pos]) {
        buffer_[pos] = std::move(grad);
    } else {
        buffer_[pos] = add_tensors(buffer_[pos].value(), grad);
    }
}

std::vector<Tensor> InputBuffer::take() {
    std::vector<Tensor> result;
    result.reserve(buffer_.size());
    for (auto& opt : buffer_) {
        if (opt) {
            result.push_back(std::move(*opt));
        } else {
            result.emplace_back();
        }
    }
    buffer_.clear();
    return result;
}

} // namespace tensorforge
