# TensorForge Architecture

This document describes the high-level structure of TensorForge, a C++20/CUDA 12 runtime for dense tensors and eager-mode autograd. The diagrams below show how the core abstractions fit together, how kernels are dispatched, how gradients flow, and how GPU memory is managed.

## Class Hierarchy

```mermaid
classDiagram
    class Tensor {
        +Storage storage_
        +size_t storage_offset_
        +Shape shape_
        +Stride stride_
        +shared_ptr~AutogradMeta~ autograd_meta_
        +data()
        +numel()
        +shape()
        +backward()
    }
    class Storage {
        +shared_ptr~StorageImpl~
        +bump_version()
    }
    class Shape {
        +vector~int64_t~ dims_
        +numel()
    }
    class Stride {
        +compute_row_major()
    }
    class Dtype {
        <<enum>>
        Float32
        Float16
        BFloat16
    }
    class Device {
        +DeviceType type
    }
    class AutogradMeta {
        +NodePtr~Node~ grad_fn_
        +Tensor grad_
    }
    class Node {
        <<abstract>>
        +apply()
    }
    class Edge {
        +NodePtr~Node~ function
        +uint32_t input_nr
    }
    class Engine {
        +execute()
    }
    Tensor --> Storage
    Tensor --> Shape
    Tensor --> Stride
    Tensor --> Dtype
    Tensor --> Device
    Tensor --> AutogradMeta
    AutogradMeta --> Node
    Node --> Edge
    Engine --> Node
```

`Tensor` is the user-facing value type. It owns metadata about shape and stride, references a `Storage` block, and optionally carries `AutogradMeta` for gradient bookkeeping. `Node` and `Edge` model the dynamic computation graph, and `Engine` traverses it during backward passes.

## Kernel Dispatch Flow

```mermaid
flowchart LR
    UserOp["Tensor::add(a, b)"] --> Check{"Same device?"}
    Check -->|CPU| CPUImpl["CPU loop"]
    Check -->|CUDA| Dispatch["dtype dispatch"]
    Dispatch --> FP32["launch_add~float~"]
    Dispatch --> FP16["launch_add~half~"]
    Dispatch --> BF16["launch_add~bfloat16~"]
    FP32 --> Kernel["elementwise_add kernel"]
    FP16 --> Kernel
    BF16 --> Kernel
```

Every operator first validates that its inputs live on the same device. CPU tensors fall back to scalar loops, while CUDA tensors route through a dtype dispatch layer that instantiates the same templated kernel for `float`, `half`, or `bfloat16`. This keeps the per-dtype code in one place and avoids duplicating kernel logic.

## Autograd Backward Execution

```mermaid
flowchart TB
    Loss["loss = model forward"] --> Backward["loss.backward()"]
    Backward --> Engine["Engine::execute"]
    Engine --> GraphTask["Build GraphTask + dependencies"]
    GraphTask --> Queue["ready_queue_"]
    Queue --> Apply["Node::apply(grads)"]
    Apply --> SaveGrads["AccumulateGrad or InputBuffer"]
    SaveGrads --> Next["Push to parents"]
    Next --> Done{"All nodes done?"}
    Done -->|No| Apply
    Done -->|Yes| End["Backward complete"]
```

Calling `backward()` creates a `GraphTask` that records which `Node` objects depend on which outputs. The engine feeds ready nodes through a work queue, invokes each `Node::apply` with incoming gradients, accumulates or buffers the results, and schedules parent nodes until the entire graph has been processed.

## Allocator and Memory Lifecycle

```mermaid
sequenceDiagram
    participant T as Tensor::empty
    participant A as StorageAllocator
    participant C as cudaMallocAsync
    participant U as User code
    T->>A: allocate(size, device, dtype)
    A->>C: cudaMallocAsync(ptr, stream)
    C-->>A: ptr
    A-->>T: Storage shared_ptr
    T-->>U: Tensor
    U->>A: free(Storage)
    A->>C: cudaFreeAsync(ptr, stream)
```

`Tensor::empty` asks the allocator for a `Storage` object, which in turn requests stream-ordered memory from CUDA via `cudaMallocAsync`. The returned raw pointer is wrapped in a reference-counted `Storage` and handed back as a `Tensor`. When the last reference disappears, the allocator releases the block with `cudaFreeAsync` on the same stream.

## Related Documentation

- [DESIGN.md](DESIGN.md): design decisions and trade-offs
- [BENCHMARKS.md](BENCHMARKS.md): performance baselines and targets (to be added)
- [CONTRIBUTING.md](CONTRIBUTING.md): contribution guidelines (to be added)
