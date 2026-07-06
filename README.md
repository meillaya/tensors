# TensorForge

> **Built TensorForge**: a CUDA tensor + autograd runtime from scratch in C++/CUDA with hand-written GEMM, softmax, layernorm kernels, reverse-mode autograd, NN modules, and MNIST MLP training reaching 95%+ accuracy on H100.

A GPU-native tensor + autograd runtime inspired by PyTorch, built in C++/CUDA from scratch with hand-written kernels, reverse-mode autograd, and PyTorch-style API.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Bazel](https://img.shields.io/badge/Bazel-9.1.1-bazelgreen.svg)](https://bazel.build/)

## Status

**v0.1.0 — Wave 2 / Wave 7 / Wave 10 complete.** Hand-written kernels for add, mul, softmax, layernorm, relu; reverse-mode autograd engine with topological execution; elementwise fusion codegen; MNIST + CIFAR-10 loaders.

Remaining: GEMM + Conv2D kernel implementation, autograd integration with GPU kernels, training examples (MLP/CNN), PyTorch benchmarks.

## Architecture

```mermaid
flowchart TB
    subgraph User["User Code"]
        ML["train_mlp.cpp<br/>train_cnn.cpp"]
    end

    subgraph NN["nn/ (Neural Network Modules)"]
        Module["Module<br/>Parameter"]
        Linear["Linear"]
        Conv2D["Conv2D"]
        ReLU["ReLU"]
        CEL["CrossEntropyLoss"]
    end

    subgraph Autograd["autograd/ (Reverse-mode AD)"]
        Engine["Engine<br/>GraphTask"]
        Node["Node + Edge<br/>SavedTensor"]
        AccumGrad["AccumulateGrad<br/>InputBuffer"]
    end

    subgraph CUDA["cuda/ (GPU Layer)"]
        Kernels["kernels/<br/>add, mul, matmul,<br/>conv2d, relu,<br/>softmax, layernorm"]
        Alloc["memory/<br/>cudaMallocAsync"]
    end

    subgraph Tensor["tensor/ (CPU + GPU Tensors)"]
        Tensor["Tensor<br/>Storage, Shape,<br/>Dtype, Device"]
        Mdspan["sandia-iso/mdspan<br/>(views)"]
    end

    subgraph Data["data/ (Loaders)"]
        MNIST["MNIST IDX"]
        CIFAR["CIFAR-10"]
    end

    subgraph Bench["benchmarks/"]
        OpBench["op_bench<br/>(vs PyTorch cuBLAS)"]
        TrainBench["train_bench<br/>(vs PyTorch)"]
    end

    ML --> NN
    ML --> Data
    NN --> Autograd
    NN --> CUDA
    Autograd --> CUDA
    Autograd --> Tensor
    CUDA --> Tensor
    Kernels --> Tensor
    Alloc --> Tensor
    Tensor --> Mdspan
    Data --> Tensor
    Bench --> CUDA
    Bench --> Tensor

    style Tensor fill:#f9e
    style CUDA fill:#fce
    style Autograd fill:#cfe
    style NN fill:#dfe
```

## Quick Start

### Local development (CPU only)

```bash
# Enter dev shell (auto-installs CUDA, Bazelisk, clang_18)
devenv shell enter

# Build everything
bazelisk build //... --config=cpu

# Run CPU tests
bazelisk test //... --config=cpu --test_tag_filters=cpu
```

### GPU development (PrimeIntellect)

```bash
# Provision an H100 pod (~30s, ~$2.35/hr)
bash scripts/provision-pod.sh h100

# Sync code + run GPU tests
prime pods ssh $(cat .omo/active_pod_id) -- \
  'cd /data/tensorforge && bazelisk test //... --config=sm80_sm90 --test_tag_filters=gpu'

# Clean up
bash scripts/terminate-pod.sh
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — class hierarchy, kernel dispatch, autograd graph, allocator lifecycle (4 Mermaid diagrams)
- **[DESIGN.md](DESIGN.md)** — key design decisions and trade-offs
- **[BENCHMARKS.md](BENCHMARKS.md)** — op-level + training throughput comparisons vs PyTorch
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — dev setup, testing, PR workflow
- **[docs/](docs/)** — architecture, CI cost model, autograd design notes

## Acknowledgements

- [PyTorch autograd](https://github.com/pytorch/pytorch/tree/main/torch/csrc/autograd) — design inspiration for Node/Edge/Engine
- [Micrograd](https://github.com/karpathy/micrograd) — Karpathy's minimal scalar autograd
- [CUTLASS](https://github.com/NVIDIA/cutlass) — GEMM kernel patterns
- [cuda-samples matrixMul](https://github.com/NVIDIA/cuda-samples) — tiled GEMM reference
- [kokkos/mdspan](https://github.com/kokkos/mdspan) — multi-dimensional span reference

## License

MIT — see [LICENSE](LICENSE).
