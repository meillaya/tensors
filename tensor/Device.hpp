#pragma once

#include <cstdint>
#include <iostream>

namespace tensorforge {

enum class DeviceType : uint8_t {
    CPU,
    CUDA,
};

struct Device {
    DeviceType type = DeviceType::CPU;
    int32_t index = 0;

    [[nodiscard]] static Device cpu() noexcept { return Device{DeviceType::CPU, 0}; }

    [[nodiscard]] static Device cuda(int32_t i = 0) noexcept { return Device{DeviceType::CUDA, i}; }

    [[nodiscard]] bool operator==(const Device& other) const noexcept {
        return type == other.type && index == other.index;
    }

    [[nodiscard]] bool operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }
};

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    switch (device.type) {
    case DeviceType::CPU:
        os << "CPU";
        break;
    case DeviceType::CUDA:
        os << "CUDA:" << device.index;
        break;
    }
    return os;
}

} // namespace tensorforge
