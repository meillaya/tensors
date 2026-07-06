"""
fusion/demo.py - relu(a*x + b) fusion vs separate-kernels demo.

Why this exists
---------------
TensorForge elementwise ops currently each launch their own CUDA kernel,
performing a global-memory read + write per op. For a chain like
relu(a*x + b) this is 4 kernel launches and 3 round-trips through DRAM.

The fusion IR collapses the chain into a single kernel: `out[i] = relu(a*x+b)`
— one read, one write, one launch. We expect ≥1.5× speedup on a large
input vector compared with the unfused baseline.

GPU benchmark deferred
----------------------
The benchmark requires:
  1. A CUDA-capable GPU (H100 / A100).
  2. nvcc + CUDA toolkit on PATH.
  3. A way to copy input data to / from device memory (currently a plain
     NumPy array; on the pod we'll use either ctypes-driven cudaMemcpy
     or cupy if it's installed by the provisioning script).

Run on the pod:

    python3 fusion/demo.py

This script scaffolds the demo so the IR / codegen modules can be
imported end-to-end and exercised by the test suite, but does NOT
execute the benchmark without a live GPU. The placeholder prints a
one-liner confirming the load path is intact.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from typing import Optional

from fusion.ir import Input, Const, Mul, Add, Relu, Expr
from fusion.codegen import compile_and_load, find_nvcc, LoadedKernel


def _has_nvcc() -> bool:
    return find_nvcc() is not None


def build_fused_expr() -> Expr:
    """The fused expression: y = relu(2*x + 3)."""
    return Relu(Add(Mul(Input("x"), Const(2.0)), Const(3.0)))


def build_fused_kernel(
    kernel_name: str = "relu_2x_3",
    arch: str = "sm_80",
    nvcc_path: Optional[str] = None,
) -> LoadedKernel:
    """Compile the fused kernel without launching it."""
    return compile_and_load(
        build_fused_expr(),
        kernel_name=kernel_name,
        arch=arch,
        nvcc_path=nvcc_path,
    )


def demo(
    n: int = 10_000_000,
    arch: str = "sm_80",
    nvcc_path: Optional[str] = None,
    force: bool = False,
) -> int:
    """Scaffolded demo entrypoint.

    Returns exit status. 0 = nothing to do (already verified locally),
    1 = nvcc not available (deferred to pod), 2 = success on pod.
    """
    nvcc = nvcc_path or find_nvcc()
    if nvcc is None and not force:
        print(
            "Fusion demo scaffolded; full benchmark requires GPU pod.\n"
            "  - On the pod with nvcc on PATH, re-run:\n"
            "      python3 fusion/demo.py\n"
            "  - Expected outcome: fused kernel achieves >=1.5x speedup\n"
            "    vs separate kernels on >=10M elements.\n"
            "  - See fusion/README.md (Fusion Demo section) for details."
        )
        return 1

    k = build_fused_kernel(nvcc_path=nvcc, arch=arch)
    print(
        f"[demo] compiled fused kernel\n"
        f"        name: {k.kernel_name}\n"
        f"        src:  {k.src_path}\n"
        f"        so:   {k.so_path}\n"
        f"        size: {k.so_path.stat().st_size} bytes\n"
        f"\n[demo] GPU benchmark deferred — requires live CUDA device."
    )
    return 2


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--n",
        type=int,
        default=10_000_000,
        help="input length (default 10M; benchmark mode only)",
    )
    parser.add_argument(
        "--arch",
        default="sm_80",
        help="target SM (sm_80 / sm_90) — benchmark mode only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="attempt compilation even when find_nvcc() is None",
    )
    args = parser.parse_args(argv)
    return demo(n=args.n, arch=args.arch, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
