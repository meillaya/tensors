"""
fusion/codegen.py - Lower IR to CUDA C++ and compile to a shared object.

The elementwise kernel template hard-codes:
  - 1D grid: 1 thread per output element
  - 256 threads per block (good for sm_80/sm_90 occupancy)
  - float32 IO (matching the IR's storage type)
  - extern "C" symbol names so ctypes can find them without name mangling

The launch wrapper takes a `cudaStream_t` so callers can plumb the
TensorForge C++ runtime stream through without a global. Both the kernel
and the launcher are compiled into a single `.so`.

Caching:
  Compiled .so artifacts live in `${TMPDIR:-/tmp}/tensorforge_fusion_cache/`,
  keyed by a SHA-256 hash of the generated source. Repeat calls with the
  same expression skip nvcc entirely.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from fusion.ir import Expr


# ---------------------------------------------------------------------------
# Kernel template
# ---------------------------------------------------------------------------


# NOTE: in C++ string formatters, literal `{` is `{{` and `}` is `}}`.
# The template uses an f-string with `.format()` to substitute three
# fields: kernel_name, kernel_name (again for the launcher), and expr.
KERNEL_TEMPLATE = """\
#include <cuda_runtime.h>
#include <cmath>

extern "C" __global__ void {kernel_name}(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    out[i] = {expr};
}}

extern "C" void launch_{kernel_name}(
    const float* in,
    float* out,
    int n,
    cudaStream_t stream
) {{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    {kernel_name}<<<blocks, threads, 0, stream>>>(in, out, n);
}}
"""


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "tensorforge_fusion_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class LoadedKernel:
    """Holds everything needed to invoke a compiled fusion kernel."""

    kernel_name: str
    src_path: Path
    so_path: Path
    lib: ctypes.CDLL
    arch: str

    def launcher(self) -> Callable:
        """Return the `launch_<name>` function with C-callable signature."""
        fn = getattr(self.lib, f"launch_{self.kernel_name}")
        fn.argtypes = [
            ctypes.c_void_p,  # const float* in
            ctypes.c_void_p,  # float* out
            ctypes.c_int,     # n
            ctypes.c_void_p,  # cudaStream_t
        ]
        fn.restype = None
        return fn


# ---------------------------------------------------------------------------
# nvcc discovery
# ---------------------------------------------------------------------------


def find_nvcc() -> Optional[str]:
    """Locate nvcc on PATH or in well-known CUDA toolkit locations."""
    found = shutil.which("nvcc")
    if found:
        return found
    for prefix in ("/usr/local/cuda/bin", "/opt/cuda/bin"):
        cand = Path(prefix) / "nvcc"
        if cand.is_file():
            return str(cand)
    return None


# ---------------------------------------------------------------------------
# Compile pipeline
# ---------------------------------------------------------------------------


def compile_and_load(
    expr: Expr,
    kernel_name: str = "fused",
    arch: str = "sm_80",
    nvcc_path: Optional[str] = None,
    force_recompile: bool = False,
    extra_include_dirs: Optional[list[str]] = None,
    extra_lib_dirs: Optional[list[str]] = None,
    allow_unsupported_compiler: bool = False,
) -> LoadedKernel:
    """Lower `expr` to a CUDA kernel and load the resulting `.so`.

    Parameters
    ----------
    expr        : the IR expression to fuse (calls `.codegen()`).
    kernel_name : symbol basename for both kernel and launcher.
    arch        : target SM, e.g. ``sm_80`` (Ampere) or ``sm_90`` (Hopper).
    nvcc_path   : explicit path to nvcc; auto-discovered if ``None``.
    force_recompile : bypass the .so cache.
    extra_include_dirs : extra ``-I`` paths (used to find ``cuda_runtime.h``
                  on systems where the toolkit is installed in non-default
                  locations, e.g. a bazel-cache-toolkit).
    extra_lib_dirs    : extra ``-L`` paths (used to find ``libcudart.so``).
    allow_unsupported_compiler : pass ``-allow-unsupported-compiler`` to nvcc.
                  Useful on dev hosts where the system gcc is newer than the
                  CUDA-supported version; should be left False on the pod.

    Raises
    ------
    RuntimeError if nvcc cannot be found or compilation fails.
    """
    nvcc = nvcc_path or find_nvcc()
    if nvcc is None:
        raise RuntimeError(
            "nvcc not found. Install CUDA toolkit or set $PATH "
            "(see scripts/provision-pod.sh for the GPU workflow)."
        )

    src = KERNEL_TEMPLATE.format(kernel_name=kernel_name, expr=expr.codegen())

    src_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cache = _cache_dir()
    so_path = cache / f"{kernel_name}_{src_hash}.so"
    cu_path = cache / f"{kernel_name}_{src_hash}.cu"

    if force_recompile and so_path.exists():
        so_path.unlink()

    if not so_path.exists():
        cu_path.write_text(src, encoding="utf-8")
        cmd = [nvcc, f"-arch={arch}", "-std=c++20", "-O3", "--shared",
               "-Xcompiler", "-fPIC", "--cudart=shared"]
        for inc in extra_include_dirs or []:
            cmd += ["-I", inc]
        for libd in extra_lib_dirs or []:
            cmd += ["-L", libd]
        if allow_unsupported_compiler:
            cmd += ["-allow-unsupported-compiler"]
        cmd += ["-o", str(so_path), str(cu_path)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                "nvcc failed (exit={code})\n"
                "  source: {cu}\n"
                "  cmd:    {cmd}\n"
                "  stderr:\n{err}".format(
                    code=result.returncode,
                    cu=cu_path,
                    cmd=" ".join(cmd),
                    err=result.stderr,
                )
            )

    lib = ctypes.CDLL(str(so_path))
    return LoadedKernel(
        kernel_name=kernel_name,
        src_path=cu_path,
        so_path=so_path,
        lib=lib,
        arch=arch,
    )


__all__ = [
    "KERNEL_TEMPLATE",
    "LoadedKernel",
    "find_nvcc",
    "compile_and_load",
]
