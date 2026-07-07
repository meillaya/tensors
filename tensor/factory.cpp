// TensorForge — factory functions (Wave 1-3, T16 extension)

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
        // FP16 bit pattern: convert through float for correctness.
        uint16_t* ptr = static_cast<uint16_t*>(data);
        for (int64_t i = 0; i < numel; ++i) {
            float v = static_cast<float>(start + i * step);
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(float));
            uint32_t sign = (bits >> 31) & 0x1;
            uint32_t exp = (bits >> 23) & 0xff;
            uint32_t mantissa = bits & 0x7fffff;
            uint32_t fp16_exp = (exp == 0) ? 0 : (exp - 127 + 15);
            if (fp16_exp >= 31) fp16_exp = 31;
            uint32_t fp16_bits = (sign << 15) | (fp16_exp << 10) | (mantissa >> 13);
            ptr[i] = static_cast<uint16_t>(fp16_bits);
        }
        break;
    }
    case Dtype::BFloat16: {
        uint16_t* ptr = static_cast<uint16_t*>(data);
        for (int64_t i = 0; i < numel; ++i) {
            float v = static_cast<float>(start + i * step);
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(float));
            // BF16 is the top 16 bits of FP32.
            ptr[i] = static_cast<uint16_t>(bits >> 16);
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

// Build a CPU staging tensor then move to target device.
Tensor build_via_cpu(Shape shape, Dtype dtype, Device target,
                      void (*cpu_fill)(void*, int64_t, Dtype)) {
    Tensor cpu = Tensor::empty(shape, dtype, Device::cpu());
    cpu_fill(cpu.data(), cpu.numel(), dtype);
    return cpu.to(target);
}

} // namespace

Tensor zeros(Shape shape, Dtype dtype, Device device) {
    return full(shape, 0.0, dtype, device);
}

Tensor ones(Shape shape, Dtype dtype, Device device) {
    return full(shape, 1.0, dtype, device);
}

Tensor full(Shape shape, double value, Dtype dtype, Device device) {
    if (device.type == DeviceType::CPU) {
        Tensor t = Tensor::empty(shape, dtype, device);
        fill_with_value(t.data(), t.numel(), value, dtype);
        return t;
    }
    // For CUDA: build on CPU then move. Avoids needing a dedicated CUDA fill
    // kernel in T16.
    Tensor cpu = Tensor::empty(shape, dtype, Device::cpu());
    fill_with_value(cpu.data(), cpu.numel(), value, dtype);
    return cpu.to(device);
}

Tensor arange(int64_t start, int64_t end, int64_t step, Dtype dtype, Device device) {
    if (step == 0) {
        throw std::invalid_argument("arange step must not be zero");
    }

    int64_t count = 0;
    if (step > 0) {
        count = (end > start) ? (end - start + step - 1) / step : 0;
    } else {
        count = (start > end) ? (start - end - step - 1) / (-step) : 0;
    }

    if (device.type == DeviceType::CPU) {
        Tensor t = Tensor::empty(Shape{count}, dtype, device);
        fill_arange(t.data(), t.numel(), start, step, dtype);
        return t;
    }

    Tensor cpu = Tensor::empty(Shape{count}, dtype, Device::cpu());
    fill_arange(cpu.data(), cpu.numel(), start, step, dtype);
    return cpu.to(device);
}

Tensor copy(const Tensor& src, Device dst_device) {
    return src.to(dst_device);
}

} // namespace tensorforge