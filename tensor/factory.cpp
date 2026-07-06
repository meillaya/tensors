#include "tensor/factory.hpp"

#include "tensor/CPUStorageAllocator.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tensorforge {

namespace {

void fill_with_value(void* data, int64_t numel, double value, Dtype dtype) {
    switch (dtype) {
    case Dtype::Float32: {
        float v = static_cast<float>(value);
        float* ptr = static_cast<float*>(data);
        std::fill(ptr, ptr + numel, v);
        break;
    }
    case Dtype::Float16: {
        // Minimal Float16 storage: reinterpret uint16_t bits as half.
        uint16_t v = static_cast<uint16_t>(static_cast<int16_t>(value));
        uint16_t* ptr = static_cast<uint16_t*>(data);
        std::fill(ptr, ptr + numel, v);
        break;
    }
    case Dtype::BFloat16: {
        uint16_t v = static_cast<uint16_t>(static_cast<int16_t>(value));
        uint16_t* ptr = static_cast<uint16_t*>(data);
        std::fill(ptr, ptr + numel, v);
        break;
    }
    case Dtype::Int32: {
        int32_t v = static_cast<int32_t>(value);
        int32_t* ptr = static_cast<int32_t*>(data);
        std::fill(ptr, ptr + numel, v);
        break;
    }
    case Dtype::Int64: {
        int64_t v = static_cast<int64_t>(value);
        int64_t* ptr = static_cast<int64_t*>(data);
        std::fill(ptr, ptr + numel, v);
        break;
    }
    case Dtype::Bool: {
        uint8_t v = value != 0.0 ? 1 : 0;
        uint8_t* ptr = static_cast<uint8_t*>(data);
        std::fill(ptr, ptr + numel, v);
        break;
    }
    }
}

void fill_arange(void* data, int64_t numel, int64_t start, int64_t step, Dtype dtype) {
    switch (dtype) {
    case Dtype::Float32: {
        float* ptr = static_cast<float*>(data);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = static_cast<float>(start + i * step);
        }
        break;
    }
    case Dtype::Float16: {
        uint16_t* ptr = static_cast<uint16_t*>(data);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = static_cast<uint16_t>(start + i * step);
        }
        break;
    }
    case Dtype::BFloat16: {
        uint16_t* ptr = static_cast<uint16_t*>(data);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = static_cast<uint16_t>(start + i * step);
        }
        break;
    }
    case Dtype::Int32: {
        int32_t* ptr = static_cast<int32_t*>(data);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = static_cast<int32_t>(start + i * step);
        }
        break;
    }
    case Dtype::Int64: {
        int64_t* ptr = static_cast<int64_t*>(data);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = start + i * step;
        }
        break;
    }
    case Dtype::Bool: {
        uint8_t* ptr = static_cast<uint8_t*>(data);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = (start + i * step) != 0 ? 1 : 0;
        }
        break;
    }
    }
}

void ensure_cpu(Device device) {
    if (device.type != DeviceType::CPU) {
        throw std::invalid_argument("factory ops only support CPU devices in Wave 2");
    }
}

} // namespace

Tensor zeros(Shape shape, Dtype dtype, Device device) {
    return full(shape, 0.0, dtype, device);
}

Tensor ones(Shape shape, Dtype dtype, Device device) {
    return full(shape, 1.0, dtype, device);
}

Tensor full(Shape shape, double value, Dtype dtype, Device device) {
    ensure_cpu(device);
    Tensor t = Tensor::empty(shape, dtype, device);
    fill_with_value(t.data(), t.numel(), value, dtype);
    return t;
}

Tensor arange(int64_t start, int64_t end, int64_t step, Dtype dtype, Device device) {
    ensure_cpu(device);
    if (step == 0) {
        throw std::invalid_argument("arange step must not be zero");
    }

    int64_t count = 0;
    if (step > 0) {
        count = (end > start) ? (end - start + step - 1) / step : 0;
    } else {
        count = (start > end) ? (start - end - step - 1) / (-step) : 0;
    }

    Tensor t = Tensor::empty(Shape{count}, dtype, device);
    fill_arange(t.data(), t.numel(), start, step, dtype);
    return t;
}

Tensor copy(const Tensor& src, Device dst_device) {
    ensure_cpu(dst_device);
    if (src.device().type != DeviceType::CPU) {
        throw std::invalid_argument("copy only supports CPU source in Wave 2");
    }

    Tensor dst = Tensor::empty(src.shape(), src.dtype(), dst_device);
    std::memcpy(dst.data(), src.data(), static_cast<size_t>(src.numel()) * dtype_size(src.dtype()));
    return dst;
}

} // namespace tensorforge
