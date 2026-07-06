# Autograd Engine Design

## Why std::shared_ptr for v1?

**Decision**: Use `std::shared_ptr<Node>` for v1. Defer a custom
`intrusive_ptr` to v2 if profiling shows it matters.

### Rationale

1. **No cycles in v1.** The autograd graph is a single DAG: forward pass
   builds edges from inputs to outputs; backward pass traverses from
   outputs to inputs. `std::shared_ptr` ownership flows along the same
   direction, so there are no reference cycles that would require
   `std::weak_ptr` breaks inside the graph itself.

2. **Standard and well-understood.** `std::shared_ptr` is part of the
   standard library, has well-defined thread-safety guarantees, and is
   familiar to all C++ developers. This reduces the learning curve and
   the surface area for subtle bugs during the initial implementation.

3. **Simpler code.** No custom reference-counting logic, no custom
   destructor wiring, no ABI concerns. The entire `IntrusivePtr.hpp`
   header is two `using` aliases.

4. **Switchable later.** All autograd code refers to `NodePtr<T>` and
   `WeakNodePtr<T>` (defined in `autograd/IntrusivePtr.hpp`), never to
   `std::shared_ptr` / `std::weak_ptr` directly. Swapping the alias to
   a custom intrusive pointer type is a single-point change that does
   not require touching any call site.

### When to revisit

- If profiling on H100 shows that `std::shared_ptr`'s atomic refcount
  increment/decrement is a measurable bottleneck in the backward pass
  (e.g., > 5% of engine time).
- If we introduce graph retention (`keep_graph = true`) with long-lived
  references that would benefit from a more compact representation.

Until then, `std::shared_ptr` is the right choice for simplicity and
correctness.

## Component Overview

```
IntrusivePtr.hpp   — NodePtr / WeakNodePtr aliases
Node.hpp           — abstract Node base + Edge struct
SavedTensor.hpp   — forward snapshot + version check
AddBackward.hpp   — concrete Node for testing (z = a + b)
AccumulateGrad.hpp — leaf sink (sequence_nr = UINT64_MAX)
InputBuffer.hpp   — accumulates grads from multiple paths
AutogradMeta.hpp  — per-Tensor autograd state (grad_fn, grad, etc.)
Engine.hpp        — topological execution via dependency counter
```

## Execution Model

1. **Dependency counting**: BFS from the root node, counting how many
   edges point to each node. A node is "ready" when its count drops to 0.
2. **Ready queue**: FIFO of nodes whose dependencies are satisfied.
3. **Per-node evaluation**: call `apply(input_grads)` → distribute output
   grads to `next_edges_` InputBuffers → decrement downstream counts.
4. **AccumulateGrad**: leaf sink with `sequence_nr = UINT64_MAX`; writes
   the accumulated gradient to `variable_.grad_`.
