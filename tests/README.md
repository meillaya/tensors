# TensorForge Test Suite

Tests use **doctest** with the following tag taxonomy:

| Tag | Meaning | Used by |
|---|---|---|
| `[cpu]` | Runs on any host, no GPU required | CI smoke, local development |
| `[gpu]` | Requires CUDA-capable GPU | PrimeIntellect pod runs |
| `[fp32]` | FP32-specific correctness | Kernel tests |
| `[fp16]` | FP16-specific correctness | Kernel tests |
| `[bf16]` | BF16-specific correctness | Kernel tests |
| `[autograd]` | Autograd integration | Backward kernel tests |
| `[benchmark]` | Performance benchmark | `bazel run` only, not `bazel test` |

## Running tests

```bash
# CPU smoke (no GPU)
bazelisk test //tests:cpu_smoke_test --config=cpu

# GPU tests (requires PrimeIntellect pod)
bazelisk test //tests:gpu_smoke_test --config=sm80_sm90

# All CPU tests
bazelisk test //... --config=cpu --test_tag_filters=cpu

# All GPU tests
bazelisk test //... --config=sm80_sm90 --test_tag_filters=gpu

# Everything except benchmarks
bazelisk test //... --test_tag_filters=-benchmark
```
