> Built TensorForge: a CUDA tensor + autograd runtime from scratch in C++/CUDA with hand-written GEMM, softmax, layernorm kernels, reverse-mode autograd, NN modules, and MNIST MLP training reaching 95%+ accuracy on H100.

# TensorForge

TensorForge is a GPU-native tensor + autograd runtime inspired by PyTorch, built in C++/CUDA from scratch with hand-written kernels, reverse-mode autograd, and PyTorch-style API.

## Status

Work in progress. The project is in early development; APIs, build targets, and directory layout are subject to change.

## Quick Start

```bash
# Enter the development shell (requires devenv)
devenv shell enter

# Build the project with Bazel
bazel build //...
```

## Documentation

Documentation will live in the `docs/` directory as it is added.

## License

TensorForge is released under the [MIT License](LICENSE).

## Acknowledgements

TensorForge is inspired by and builds on ideas from:

- [PyTorch](https://pytorch.org/)
- [Micrograd](https://github.com/karpathy/micrograd)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Bazel](https://bazel.build/)
- [rules_cuda](https://github.com/bazel-contrib/rules_cuda)
