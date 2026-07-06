// TensorForge - Build Validation Wave 0.5
// Minimal CUDA kernel + host code to prove Bazel + rules_cuda + sm_80/sm_90 toolchain.
//
// C++20 ONLY. Do NOT use std::mdspan, std::expected, deducing-this, or any C++23
// feature - nvcc in CUDA 12.x only accepts --std=c++20.
//
// IMPORTANT: include <cstdint> / <cstddef> BEFORE <cuda_runtime.h>. nvcc on
// CUDA 12.x + libstdc++ 12 has a `__noinline__` macro conflict between CUDA's
// host_defines.h and libstdc++'s __attribute__((__noinline__)).

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include "hello.cuh"

namespace tensorforge {

__device__ void kernel_says() {
    // Thread-local print from inside a CUDA kernel.
    printf("TensorForge build OK (from __global__ thread %d)\n",
           static_cast<int>(threadIdx.x));
}

// Kernel launcher - defined in the .cu translation unit so the
// triple-bracket `<<<...>>>` syntax is only ever seen by nvcc, not by the
// host cc_toolchain (gcc) that compiles main.cc.
__global__ void hello_kernel() { kernel_says(); }

void launch_hello_kernel() {
    hello_kernel<<<1, 1>>>();
}

}  // namespace tensorforge
