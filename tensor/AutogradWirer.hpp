#pragma once

// TensorForge - Autograd wirer registration (Tensor <-> Autograd bridge)
//
// Tensor::operator+, Tensor::operator*, and the activations/matmul/softmax/
// layernorm/log/elementwise methods live in tensor/Tensor.cpp, but
// tensor_core must not depend on autograd_core (the reverse dep already
// exists). To wire the various *Backward Nodes from autograd without a
// library-level cycle, autograd/TensorAutograd.cpp registers C-style
// function pointers via the API below. tensor/Tensor.cpp calls them
// after the kernel dispatch when an input requires_grad.
//
// If autograd_core is not linked into the binary, the function pointers
// stay null and the operations simply skip the autograd wiring - the
// forward pass is unchanged, only backward propagation is unavailable.
//
// Each wirer has a signature matching the operator it wraps. The binary
// wirer signature is (out, a, b); the unary wirer signature is (out, x).
// Unary wirers are reused for all elementwise unary ops that share the
// same backward (relu/sigmoid/tanh/leaky_relu/log).

#include "tensor/Tensor.hpp"

namespace tensorforge {

// Binary op wirer: invoked from operator+/operator*/matmul after the
// kernel call, only when the wirer is non-null and at least one input
// requires_grad. `out` already has its kernel output filled in.
using AutogradWirerFn = void (*)(Tensor& out, const Tensor& a, const Tensor& b);

// Unary op wirer: invoked from relu/sigmoid/tanh/leaky_relu/log after
// the kernel call, only when the wirer is non-null and the input
// requires_grad. `out` already has its kernel output filled in.
using AutogradUnaryWirerFn = void (*)(Tensor& out, const Tensor& x);

// Softmax / log_softmax wirer: also carries the dim so the backward
// Node can sum along the right axis.
using AutogradSoftmaxWirerFn = void (*)(Tensor& out, const Tensor& x, int64_t dim);

// Layer-norm wirer: needs gamma/beta/eps because the saved Tensor
// metadata (shape, device) lives in those args.
using AutogradLayerNormWirerFn = void (*)(Tensor& out, const Tensor& x,
                                          const Tensor& gamma,
                                          const Tensor& beta, float eps);

// Reduction wirer (sum): carries dim and keepdim so the backward Node
// can broadcast the upstream gradient back to the input shape.
using AutogradReduceWirerFn = void (*)(Tensor& out, const Tensor& x,
                                        int64_t dim, bool keepdim);

void register_add_wirer(AutogradWirerFn fn);
void register_mul_wirer(AutogradWirerFn fn);
void register_matmul_wirer(AutogradWirerFn fn);

void register_relu_wirer(AutogradUnaryWirerFn fn);
void register_sigmoid_wirer(AutogradUnaryWirerFn fn);
void register_tanh_wirer(AutogradUnaryWirerFn fn);
void register_leaky_relu_wirer(AutogradUnaryWirerFn fn);
void register_log_wirer(AutogradUnaryWirerFn fn);

void register_softmax_wirer(AutogradSoftmaxWirerFn fn);
void register_log_softmax_wirer(AutogradSoftmaxWirerFn fn);
void register_layernorm_wirer(AutogradLayerNormWirerFn fn);
void register_sum_wirer(AutogradReduceWirerFn fn);

// Idempotent toggle that registers all of the above wirers at once.
// Examples / tests should call this from main() so the registrar
// inside TensorAutograd.cpp cannot be GC'd by the linker. Safe to
// call multiple times.
void init_tensor_autograd();

} // namespace tensorforge
