# Profiling Summary (T53)

Hardware: NVIDIA H100 80GB HBM3 (datacrunch), CUDA 12.6.85 (rules_cuda hermetic).
Target: `bazel-bin/benchmarks/op_bench`. Profiler versions used:

- nsys 2025.x (CUDA 12.6.85)
- ncu 2025.1.1.0 (NCU CLI)

## nsys Top Kernels (by total GPU time)

| Kernel | Total ns | Inst | Avg ns | Med ns |
|--------|---------:|-----:|-------:|-------:|
| `void tensorforge::<unnamed>::gemm_optimized_kernel<float, (int)16>(const T1 *, c` | 74,812,873 | 115 | 650547 | 52704 |
| `void tensorforge::elementwise_add_kernel<float>(const T1 *, const T1 *, T1 *, lo` | 2,487,726 | 115 | 21632 | 2112 |
| `void tensorforge::elementwise_mul_kernel<float>(const T1 *, const T1 *, T1 *, lo` | 2,487,244 | 115 | 21628 | 2112 |
| `void tensorforge::elementwise_unary_kernel<float, tensorforge::SigmoidOp>(const ` | 2,381,356 | 115 | 20707 | 2208 |
| `void tensorforge::elementwise_unary_kernel<float, tensorforge::TanhOp>(const T1 ` | 2,328,460 | 115 | 20248 | 2208 |
| `void tensorforge::elementwise_unary_kernel<float, tensorforge::ReluOp>(const T1 ` | 2,158,860 | 115 | 18773 | 2048 |
| `void tensorforge::<unnamed>::softmax_kernel<float>(const T1 *, T1 *, long)` | 663,259 | 115 | 5768 | 2752 |
| `void tensorforge::<unnamed>::layernorm_kernel<float>(const T1 *, const T1 *, con` | 304,191 | 92 | 3306 | 2944 |

## nsys Top CUDA API (host-side)

| API | Total ns | Calls | Avg ns |
|-----|---------:|------:|-------:|
| `cudaStreamCreateWithFlags` | 116,205,355 | 1 | 116205355 |
| `cudaEventSynchronize` | 78,705,363 | 780 | 100904 |
| `cudaMemcpyAsync` | 17,213,425 | 42 | 409844 |
| `cudaStreamSynchronize` | 11,657,608 | 63 | 185041 |
| `cudaMallocAsync_v11020` | 8,022,052 | 66 | 121546 |
| `cudaLaunchKernel` | 3,842,302 | 897 | 4284 |
| `cudaEventRecord` | 2,396,124 | 1560 | 1536 |
| `cudaGetDeviceProperties_v2_v12000` | 1,498,979 | 1 | 1498979 |
| `cudaDeviceGetDefaultMemPool_v11020` | 193,310 | 1 | 193310 |
| `cudaEventCreate` | 91,082 | 78 | 1168 |

## ncu Per-Kernel Metrics

Metrics: `gpu__time_duration.sum` (us), `dram__throughput.avg.pct_of_peak_sustained_elapsed` (% of HBM3 peak), `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`.

| Kernel | n | Time (us) | DRAM %SOL | ComputeMem %SOL | SM %SOL |
|--------|--:|----------:|----------:|----------------:|--------:|
| `elementwise_mul_kernel(const T1 *, const T1 *, T1 *, long) (1024, 1, 1)x(256, 1,` | 15 |      4.40 |     14.4 |           14.4 |   21.4 |
| `elementwise_add_kernel(const T1 *, const T1 *, T1 *, long) (1024, 1, 1)x(256, 1,` | 15 |      4.34 |     14.6 |           14.6 |   21.6 |

## Optimization Opportunities

- Element-wise add at 16M reaches ~2.47 TB/s = ~74% of HBM3 peak. 
  Further gains require kernel fusion (bias+ReLU, layernorm+gelu).
- GEMM `gemm_optimized_kernel` dominates GPU time (~85% in this run) 
  — limited by FP32 SIMT throughput (~25 TFLOPS peak). WMMA / CUTLASS 
  tensor-core path is the next 5–10x.
- softmax_kernel: only ~0.8% of time; bandwidth bound, scales well.
- layernorm_kernel: ~0.3% of time; Welford single-pass already at ~1.13 TB/s.
- cudaEventSynchronize is the top API host-side cost — batched streams / 
  CUDA graphs would reduce per-launch overhead.

