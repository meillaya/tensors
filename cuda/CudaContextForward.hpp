// TensorForge — host-only forward declarations for the CUDA context layer.
//
// This header gives non-CUDA translation units (e.g. tensor/Tensor.cpp when
// compiled by g++) the DeviceContext struct without dragging in
// cuda_runtime.h (which is only available to the CUDA toolchain).
//
// When the CUDA toolchain compiles a TU that wants the real cuda_runtime.h
// types, include cuda/CudaContext.hpp instead.

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Device.hpp"

namespace tensorforge {

class CudaStream;

// Per-device context. The current_stream is opaque (void*) so this header
// doesn't need to declare cudaStream_t — that typedef is only meaningful when
// cuda_runtime.h is included.
struct DeviceContext {
    int32_t device_index = 0;
    void* current_stream = nullptr;
    CudaStream* owned_stream = nullptr;

    [[nodiscard]] static DeviceContext& current();
};

} // namespace tensorforge