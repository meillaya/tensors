# Contributing to TensorForge

Thanks for your interest in contributing! This guide covers development setup, testing, and submission workflow.

## Development Environment

TensorForge uses [devenv](https://devenv.sh/) for reproducible dev environments.

### Prerequisites
- Nix package manager
- devenv (installed via `nix profile install nixpkgs#devenv`)
- Git

### Setup

```bash
# Clone the repo
git clone https://github.com/<user>/tensorforge
cd tensorforge

# Enter the dev shell (auto-installs CUDA 12.6, Bazelisk, clang_18, etc.)
devenv shell enter

# Verify toolchain
nvcc --version  # Should show CUDA 12.6.x
bazelisk --version  # Should show Bazel 9.1.1
clang++ --version  # Should show clang 18.x
```

### On a machine WITHOUT Nix

If you can't use devenv.nix, the same tools can be installed manually:

```bash
# Bazelisk (downloads Bazel on demand)
curl -fsSL -o /usr/local/bin/bazelisk \
  https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
chmod +x /usr/local/bin/bazelisk

# CUDA toolkit (download from NVIDIA)
# See https://developer.nvidia.com/cuda-downloads
```

## Building & Testing

### CPU-only (no GPU required)

```bash
# Build everything
bazelisk build //... --config=cpu

# Run all CPU tests
bazelisk test //... --config=cpu --test_tag_filters=cpu
```

### GPU (requires PrimeIntellect pod)

TensorForge's GPU tests need an NVIDIA GPU. Since this project uses PrimeIntellect for development:

```bash
# Provision a pod (creates + waits for ready)
bash scripts/provision-pod.sh h100  # or a100

# Sync code + run GPU tests
prime pods ssh $(cat .omo/active_pod_id) -- \
  'cd /data/tensorforge && bazelisk test //... --config=sm80_sm90 --test_tag_filters=gpu'

# Clean up
bash scripts/terminate-pod.sh
```

See `docs/primeintellect.md` (T5) for detailed pod management.

## Code Style

TensorForge uses `.clang-format` (LLVM-based, 4-space indent). Run before committing:

```bash
clang-format --style=file -i <files>
```

Tests use [doctest](https://github.com/doctest/doctest) with tag taxonomy:
- `[cpu]` — runs on any host
- `[gpu]` — requires NVIDIA GPU
- `[fp32]`, `[fp16]`, `[bf16]` — dtype-specific
- `[autograd]` — autograd integration
- `[benchmark]` — performance, not for CI

## Adding a New Op

1. Add forward kernel to `cuda/kernels/<op>.cu` (templated on dtype)
2. Add CPU implementation to `tensor/<op>.cpp`
3. Add Tensor method `Tensor::<op>(other)` in `tensor/Tensor.hpp`
4. Add autograd backward Node to `autograd/ops/<op>_backward.hpp`
5. Wire backward in Tensor method when `requires_grad` is true
6. Write tests in `tests/<op>_test.cpp` with `[cpu][gpu][fp32][fp16][bf16]` tags
7. Verify gradient check via finite differences (tolerance 1e-3 for FP32, 1e-2 for FP16/BF16)

## Adding a New NN Module

1. Add header to `nn/<Module>.hpp` and impl to `nn/<Module>.cpp`
2. Inherit from `nn::Module`
3. Register parameters in constructor
4. Implement `forward(Tensor x)`
5. Add to `nn/BUILD.bazel` `cc_library` + `cc_test`
6. Write tests in `nn/<Module>_test.cpp`

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(<scope>): <description>` — new feature
- `fix(<scope>): <description>` — bug fix
- `chore(<scope>): <description>` — maintenance, deps
- `docs(<scope>): <description>` — docs only
- `test(<scope>): <description>` — tests only
- `bench(<scope>): <description>` — benchmarks
- `build(<scope>): <description>` — build system
- `ci(<scope>): <description>` — CI changes

Examples:
- `feat(cuda/kernels): tiled 16x16 GEMM with bank-conflict padding`
- `fix(autograd): handle empty grad_fn in Engine::execute`
- `docs(arch): update Mermaid classDiagram with new Parameter type`

## Pull Requests

1. Fork the repo, create a feature branch: `git checkout -b feat/my-feature`
2. Make your changes, commit with conventional messages
3. Add tests for new functionality
4. Verify all tests pass: `bazelisk test //... --config=cpu --test_tag_filters=cpu`
5. Push and open a PR; CI runs CPU smoke + GPU tests
6. Ensure review approval before merge

## License

By contributing, you agree that your contributions will be licensed under MIT.
