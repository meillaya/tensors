// TensorForge - Build Validation Wave 0.5 host main().
//
// Launches a trivial kernel and prints 'TensorForge build OK' from host code.
//
// The kernel launcher (`<<<...>>>`) lives in hello.cu, so the host compiler
// (gcc) that compiles this file never has to parse the triple-bracket syntax.

#include <cstddef>
#include <cstdio>

#include <cuda_runtime.h>

#include "hello.cuh"

int main(int /*argc*/, char** /*argv*/) {
    std::printf("TensorForge build OK (from main() on host)\n");
    std::fflush(stdout);
    tensorforge::launch_hello_kernel();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
