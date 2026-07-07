// TensorForge — GEMM kernel header (Wave 5 / T22-T24)
//
// Three GEMM implementations, all templated on dtype (FP32/FP16/BF16) with
// FP32 accumulation:
//
//   T22: gemm_naive           — one thread per output, no shared memory
//   T23: gemm_tiled_16x16     — 16x16 block tile + bank-padded shared mem
//   T24: gemm_optimized       — vectorized loads + double-buffered shared mem
//
// All kernels compute row-major C[M,N] = A[M,K] @ B[K,N]. Bounds checks are
// present everywhere so non-multiples of 16 still work (slower path on
// boundaries).

#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor/Dtype.hpp"

namespace tensorforge {

// Computes C = A @ B for row-major 2D matrices. A, B, C must all live on the
// same device and have matching dtype. The accumulator is always FP32 for
// the FP16/BF16 paths so we don't lose precision before the final cast.
//
// Shapes:
//   A: [M, K]   (rows: M, cols: K)
//   B: [K, N]
//   C: [M, N]
//
// `stream` may be nullptr for the default stream (legacy behavior).
void launch_gemm_naive(const void* A, const void* B, void* C,
                       int64_t M, int64_t N, int64_t K,
                       Dtype dtype, void* stream);

// Same contract as launch_gemm_naive, but the kernel uses a 16x16 thread
// block tile per output with bank-padded shared memory (As[16][17],
// Bs[16][17]). The "+1" padding breaks 32-way bank conflicts on the
// shared-memory inner-product step.
void launch_gemm_tiled_16x16(const void* A, const void* B, void* C,
                              int64_t M, int64_t N, int64_t K,
                              Dtype dtype, void* stream);

// Same contract as launch_gemm_tiled_16x16, but the kernel additionally
// uses vectorized global-memory loads (float4 / half4 / bfloat164) and
// double-buffered shared memory (As[2][16][17], Bs[2][16][17]) so the
// cooperative load of the next tile overlaps the compute of the current
// tile. Falls back to scalar loads on the boundary / non-vector-aligned
// paths.
void launch_gemm_optimized(const void* A, const void* B, void* C,
                            int64_t M, int64_t N, int64_t K,
                            Dtype dtype, void* stream);

}  // namespace tensorforge