#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tensor/CPUStorageAllocator.hpp"
#include "tensor/Storage.hpp"

using tensorforge::CPUStorageAllocator;
using tensorforge::Device;
using tensorforge::Dtype;
using tensorforge::Storage;

TEST_CASE("Storage allocation and alignment") {
    Storage s = CPUStorageAllocator::instance().allocate(128, Device::cpu(), Dtype::Float32);
    CHECK(s != nullptr);
    CHECK(s->data() != nullptr);
    CHECK(s->size_bytes() == 128);
    CHECK(s->device() == Device::cpu());
    CHECK(s->dtype() == Dtype::Float32);

    const uintptr_t addr = reinterpret_cast<uintptr_t>(s->data());
    CHECK(addr % 64 == 0);
}

TEST_CASE("Storage version counter") {
    Storage s = CPUStorageAllocator::instance().allocate(64, Device::cpu(), Dtype::Int64);
    CHECK(s->current_version() == 0);
    CHECK(s->bump_version() == 1);
    CHECK(s->current_version() == 1);
    CHECK(s->bump_version() == 2);
    CHECK(s->current_version() == 2);
}

TEST_CASE("Storage shared ownership") {
    Storage s1 = CPUStorageAllocator::instance().allocate(64, Device::cpu(), Dtype::Float32);
    CHECK(s1.use_count() == 1);
    {
        Storage s2 = s1;
        CHECK(s1.use_count() == 2);
        CHECK(s2.use_count() == 2);
    }
    CHECK(s1.use_count() == 1);
}

TEST_CASE("Storage allocator free") {
    Storage s = CPUStorageAllocator::instance().allocate(64, Device::cpu(), Dtype::Bool);
    CHECK(s->data() != nullptr);
    CPUStorageAllocator::instance().free(s);
    CHECK(s->data() == nullptr);
}
