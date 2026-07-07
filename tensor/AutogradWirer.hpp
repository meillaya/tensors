#pragma once

// TensorForge — Autograd wirer registration (Tensor ↔ Autograd bridge)
//
// Tensor::operator+ and Tensor::operator* live in tensor/Tensor.cpp, but
// tensor_core must not depend on autograd_core (the reverse dep already
// exists). To wire AddBackward/MulBackward from autograd without a
// library-level cycle, autograd/TensorAutograd.cpp registers C-style
// function pointers via the API below. tensor/Tensor.cpp calls them after
// the kernel dispatch when an input requires_grad.
//
// If autograd_core is not linked into the binary, the function pointers
// stay null and operator+/operator* simply skip the autograd wiring — the
// forward pass is unchanged, only backward propagation is unavailable.

#include "tensor/Tensor.hpp"

namespace tensorforge {

// Wirer signature: invoked from operator+/operator* after the kernel call,
// only when the wirer is non-null and at least one input requires_grad.
// `out` already has its kernel output filled in; the wirer mutates out's
// autograd metadata to attach a grad_fn.
using AutogradWirerFn = void (*)(Tensor& out, const Tensor& a, const Tensor& b);

void register_add_wirer(AutogradWirerFn fn);
void register_mul_wirer(AutogradWirerFn fn);

} // namespace tensorforge