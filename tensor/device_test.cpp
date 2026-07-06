#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/Device.hpp"

using tensorforge::Device;
using tensorforge::DeviceType;

TEST_CASE("Device CPU factory") {
    Device d = Device::cpu();
    CHECK(d.type == DeviceType::CPU);
    CHECK(d.index == 0);
    CHECK(d == Device{DeviceType::CPU, 0});
}

TEST_CASE("Device CUDA factory") {
    Device d0 = Device::cuda();
    CHECK(d0.type == DeviceType::CUDA);
    CHECK(d0.index == 0);

    Device d3 = Device::cuda(3);
    CHECK(d3.type == DeviceType::CUDA);
    CHECK(d3.index == 3);
    CHECK(d3 != d0);
}

TEST_CASE("Device equality") {
    CHECK(Device::cpu() == Device::cpu());
    CHECK(Device::cuda(1) == Device::cuda(1));
    CHECK(Device::cpu() != Device::cuda());
    CHECK(Device::cuda(0) != Device::cuda(1));
}
