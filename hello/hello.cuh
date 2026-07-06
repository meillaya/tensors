// TensorForge - Build Validation Wave 0.5
// Header for the minimal CUDA hello world.

#pragma once

namespace tensorforge {

// Marked __device__ so it's callable from a __global__ launcher in this TU.
// Implemented in hello.cu.
__device__ void kernel_says();

// Host-callable launcher that emits the kernel grid. Keeping this in the .cu
// file means the triple-bracket `<<<...>>>` syntax is only parsed by nvcc.
void launch_hello_kernel();

}  // namespace tensorforge
