# fusion/

A Python IR + .cu codegen system for **elementwise kernel fusion** in TensorForge.

This directory is pure Python (no C++/CUDA runtime touched). It produces
CUDA `.cu` source from a small AST-shaped IR, compiles the result with
`nvcc`, and loads the resulting shared object via `ctypes`. The goal is
to demonstrate that four TensorForge elementwise ops (mul, add, relu)
compose into a *single* kernel launch — instead of four separate kernel
launches with their corresponding global-memory round-trips.

## Status

| Step | Status |
|---|---|
| IR (`fusion/ir.py`) | done — T41 |
| Codegen + nvcc (`fusion/codegen.py`) | done — T42 |
| Demo scaffolding (`fusion/demo.py`) | done — T43 |
| GPU benchmark | deferred — needs GPU pod |

## Running locally (no GPU)

```bash
# IR + codegen module tests
pytest fusion/ -v

# Demo scaffold prints "deferred to pod" and exits 1
python3 fusion/demo.py
```

When `find_nvcc()` returns `None`, all `nvcc`-gated pytest cases skip;
the rest pass on a plain interpreter. On a PrimeIntellect pod (with
CUDA toolkit installed), the `TestCompileAndLoad` suite compiles and
loads a real `.so` and pytest reports the previously skipped 4 tests
as `PASSED`.

## IR design

Every node descends from `fusion.ir.Expr` and implements `codegen() -> str`
that returns a C++ expression fragment. Composition is bottom-up:

```python
from fusion.ir import Input, Const, Mul, Add, Relu

expr = Relu(Add(Mul(Input("x"), Const(2.0)), Const(3.0)))
# describes y = relu(2*x + 3)

print(expr.codegen())
# fmaxf(((x * 2.000000f) + 3.000000f), 0.0f)
```

Nodes:

| Class  | Purpose | C++ lowering |
|--------|---------|--------------|
| `Input` | scalar input `x` | `x` |
| `Const` | float literal | `2.000000f` |
| `Mul`   | `a * b`         | `(a * b)` |
| `Add`   | `a + b`         | `(a + b)` |
| `Relu`  | `max(x, 0)`     | `fmaxf(x, 0.0f)` |
| `Sigmoid` | `1/(1+exp(-x))` | `(1.0f / (1.0f + expf(-x)))` |
| `Tanh`  | `tanh(x)`       | `tanhf(x)` |

Design choices:

1. **`Const.__post_init__` normalises to float.** `Const(3)` (int) and
   `Const(3.0)` (float) both lower to `3.000000f`. This matches the
   kernel's `float x` storage type and avoids nvcc's
   "double-to-float conversion" warning under `-Wconversion`.
2. **Parenthesisation is intentionally loud.** `Mul` and `Add` always wrap
   their children in `( )`. This produces noisier strings, but the
   output is unambiguous when read by eye and matches the algebraic
   precedence visually.
3. **Single-input only.** Kernels generated here take one input buffer
   and emit one output buffer. Multi-input fusion would require a
   richer IR (different `Input` identifiers, multiple loads per thread)
   — left as a v2 extension.

## Codegen approach

`fusion.codegen.compile_and_load(expr, kernel_name)`:

1. Wraps `expr.codegen()` in a CUDA kernel template (1 thread per
   output element, 1D grid, fused `float x = in[i]; out[i] = <expr>;`).
2. Hashes the source with SHA-256; on cache hit, skips recompile.
3. On miss, writes `.cu`, invokes `nvcc -arch=sm_80 -O3 --shared`, and
   loads the resulting `.so` via `ctypes.CDLL`.
4. Returns the loaded library; the caller is responsible for building a
   Python-callable wrapper that translates a NumPy array + raw pointer
   into a `cudaMemcpy` + kernel launch.

Caching lives in `${TMPDIR:-/tmp}/tensorforge_fusion_cache/`.

## Fusion Demo

The full `relu(a*x + b)` fusion demo requires a GPU pod with nvcc
available in `$PATH`. After re-provisioning a PrimeIntellect pod
(see `scripts/provision-pod.sh`):

```bash
# On the pod, with CUDA toolkit installed:
python3 fusion/demo.py
```

Expected outcome: fused kernel achieves **≥1.5× speedup** vs separate
`mul`, `add`, `relu` kernels on a large input vector (≥10M elements),
measured by `cudaEvent_t` round-trip time end-to-end.

On this dev machine (no GPU + GCC too new for CUDA 12.6), `demo.py`
prints a clear "deferred to pod" message and returns exit 1. The
scaffolding is wired end-to-end so that the real benchmark on a pod
is just `python3 fusion/demo.py --arch=sm_90` with nothing else to
change.
