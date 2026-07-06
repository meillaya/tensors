# TensorForge Design Decisions

This document records the major engineering decisions made while building TensorForge. Each entry states what was chosen, why it was chosen, what alternatives were considered, and what trade-offs the choice entails.

## 1. C++20 (not C++23)

**Decision:** Use C++20 as the single C++ standard for the entire codebase.

**Rationale:** CUDA 12's `nvcc` only accepts `--std=c++20`. C++23 support lands in CUDA 13.3 and later, which is not available in our target environment. Restricting the project to C++20 keeps the host compiler, the device compiler, and all third-party dependencies on the same language version.

**Alternatives Considered:**

- (a) Split compilation: use C++23 in `.cpp` files and C++20 in `.cu` files. This adds interface complexity and forces any shared headers to remain C++20 anyway.
- (b) Upgrade to CUDA 13.3+. This is not practical for the current environment and would raise the minimum supported driver version.

**Trade-offs:** We lose `std::mdspan`, `std::expected`, and deducing `this`. In exchange, we gain a simpler build system and a single language baseline across host and device code.

## 2. cudaMallocAsync for v1 (no custom caching allocator)

**Decision:** Use `cudaMallocAsync`, the stream-ordered allocator introduced in CUDA 11.2, for all GPU allocations.

**Rationale:** PyTorch's custom caching allocator is roughly 5,391 lines of code and historically took an estimated one to two months to build and stabilize. A production allocator is out of scope for v1, and `cudaMallocAsync` already provides stream ordering and reasonable performance.

**Alternatives Considered:**

- (a) Build a custom caching allocator from scratch. This would improve performance but consume significant schedule time.
- (b) Use plain `cudaMalloc` / `cudaFree`. This avoids the CUDA 11.2 requirement but loses stream ordering and is generally slower due to synchronization.

**Trade-offs:** We gain four to six weeks of development time and keep the memory layer small. We lose some performance optimization opportunities that a caching allocator would provide.

## 3. im2col + GEMM for Conv2D (not direct convolution)

**Decision:** Implement Conv2D by expanding the image into columns with `im2col`, then reuse the existing tiled GEMM kernel.

**Rationale:** `im2col` is straightforward to implement correctly. Direct convolution requires careful sliding-window indexing, boundary handling, and shared-memory layout tuning to match the same level of correctness. Reusing GEMM also means Conv2D benefits from improvements to the matrix multiply path.

**Alternatives Considered:**

- (a) Direct shared-memory convolution. This avoids the memory expansion but is harder to get right.
- (b) FFT-based convolution. This is asymptotically attractive for large kernels but adds significant complexity and memory for intermediate transforms.

**Trade-offs:** `im2col` consumes extra memory for the expanded matrix. Direct convolution remains a stretch goal for a later milestone.

## 4. Linked `Node` autograd (not flat arena)

**Decision:** Use a PyTorch-style hierarchy of `Node` objects linked by `Edge` records, with `std::shared_ptr` for lifetime management in v1.

**Rationale:** The graph is built eagerly during the forward pass, so reference counting solves lifetime naturally. A linked design is easier to reason about than a flat arena and matches the shape of the dynamic computation graph.

**Alternatives Considered:**

- (a) Flat arena (JAX-style). This can be more cache-friendly but requires tracing the whole program and is a poor fit for eager execution.
- (b) `intrusive_ptr` (PyTorch production). This avoids the size overhead of `shared_ptr` control blocks but adds manual reference-counting boilerplate.

**Trade-offs:** There is slight indirection overhead from heap-allocated nodes and `shared_ptr`. If profiling shows this matters, we can migrate to `intrusive_ptr` in v2.

## 5. One CUDA stream per backward call (v1)

**Decision:** Run each backward pass on a single CUDA stream.

**Rationale:** Single-stream execution makes correctness easier to reason about because every kernel in the backward graph is serialized. Per-operator stream recording introduces complex synchronization edges, and production frameworks still find bugs in that area.

**Alternatives Considered:**

- (a) Per-operator stream recording from the start. This could increase concurrency but complicates dependency tracking.
- (b) Full CUDA Graph capture. This is attractive for repeated training steps but requires the entire backward graph to be replayable.

**Trade-offs:** We leave potential concurrency on the table. Multi-stream and graph capture are deferred to v2.

## 6. AST-based Python codegen for fusion (v1)

**Decision:** Implement kernel fusion with a Python IR that emits a single fused `.cu` file, compiled via `subprocess` and loaded with `ctypes`.

**Rationale:** This is the minimum viable approach to fusion. It demonstrably outperforms launching separate kernels for simple pointwise chains, and it avoids building a full compiler stack.

**Alternatives Considered:**

- (a) Template metaprogramming at C++ compile time. This fuses at build time but cannot fuse arbitrary user expressions discovered at runtime.
- (b) Full compiler stack (Triton-style). This is powerful but far beyond v1 scope.
- (c) TorchInductor-style codegen. This is closer to our approach but still requires a much larger IR and scheduling layer.

**Trade-offs:** There is a compile-time cost the first time a fused kernel is encountered. Fusion is limited to pointwise chains in v1; richer fusion patterns are deferred to v2.

## 7. Templated dtype dispatch in kernels (not in `Node::backward`)

**Decision:** Keep kernels templated on scalar type, while `Node::backward` remains dtype-agnostic.

**Rationale:** This mirrors PyTorch's separation of concerns. Type promotion and dispatch happen at the operator layer, while the backward function only needs to know how to combine gradients. It also avoids instantiating the entire autograd graph for every dtype.

**Alternatives Considered:**

- (a) Template `Node::backward`. This would pull dtype information into the graph and increase binary size.
- (b) ATen-style dispatch macros. These are expressive but introduce a heavy macro layer that complicates debugging.

**Trade-offs:** There is some duplication in dtype handling logic at the dispatch layer, but the autograd core stays clean and small.

## 8. Hand-rolled GEMM won't match cuBLAS (realistic 10% target)

**Decision:** Document that the hand-rolled tiled GEMM will be roughly 5-10x slower than cuBLAS, and frame benchmarks as an optimization journey rather than a claim of parity.

**Rationale:** cuBLAS uses tensor cores, sophisticated scheduling, and vendor-tuned assembly. A hand-written FP32 kernel uses plain FMA instructions. Reaching production parity would require CUTLASS or a similar library.

**Alternatives Considered:**

- (a) Use cuBLAS directly. This would give parity immediately but teaches less about how GEMM works.
- (b) Integrate CUTLASS from the start. This is the right production choice but is too large for v1.

**Trade-offs:** Benchmarks must be presented honestly. CUTLASS integration is a stretch goal, and the hand-rolled kernel is treated as a learning baseline.

## 9. rules_cuda pinned to commit (not release tag)

**Decision:** Pin `rules_cuda` to commit `345dd02d2b64b8a6c138800da3180c9278417ec2` using `git_override`.

**Rationale:** Fixes for multi-version and multi-architecture builds landed after the v0.3.0 release tag. The chosen commit contains those fixes while still being a stable single revision.

**Alternatives Considered:**

- (a) Pin to the v0.3.0 release tag. This is more conventional but lacks the post-release fixes we need.
- (b) Float on the latest commit. This would pick up future fixes automatically but also risks breakage from upstream changes.

**Trade-offs:** We must manually bump the pinned commit to receive future fixes. For now, this is safer than either the release tag or a floating dependency.
