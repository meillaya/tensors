"""
fusion/test_codegen.py - compile-and-load tests for fusion/codegen.py.

Most tests skip when nvcc is unavailable; on a GPU pod with
CUDA toolkit installed they execute the full nvcc round-trip.
"""

from __future__ import annotations

import ctypes
import shutil

import pytest

from fusion.codegen import (
    KERNEL_TEMPLATE,
    LoadedKernel,
    _cache_dir,
    compile_and_load,
    find_nvcc,
)
from fusion.ir import Input, Const, Mul, Add, Relu, Sigmoid, Tanh


# ---------------------------------------------------------------------------
# Tests that do NOT require nvcc
# ---------------------------------------------------------------------------


class TestTemplate:
    def test_template_has_kernel_and_launcher(self) -> None:
        assert "__global__ void" in KERNEL_TEMPLATE
        assert "extern \"C\"" in KERNEL_TEMPLATE
        assert "launch_" in KERNEL_TEMPLATE
        assert "<<<" in KERNEL_TEMPLATE and ">>>" in KERNEL_TEMPLATE

    def test_template_uses_cmath(self) -> None:
        # Needed for fmaxf / expf / tanhf intrinsics in unary nodes.
        assert "#include <cmath>" in KERNEL_TEMPLATE

    def test_template_format_substitutes_all_fields(self) -> None:
        out = KERNEL_TEMPLATE.format(
            kernel_name="mykernel", expr="x + 1.0f"
        )
        assert "mykernel" in out
        assert "x + 1.0f" in out
        # No leftover f-string placeholders (literal {} braces from format
        # escaping were doubled to {{ and }} in the template, so single
        # braces here only appear as C++ statement blocks).
        assert "{kernel_name}" not in out
        assert "{expr}" not in out


class TestCache:
    def test_cache_dir_exists(self) -> None:
        assert _cache_dir().is_dir()


class TestFindNvcc:
    def test_find_nvcc_returns_str_or_none(self) -> None:
        # Either path or None is acceptable; on a pod it's a path, on this
        # dev box without a toolkit it may be None.
        result = find_nvcc()
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests that DO require nvcc (skipped without it)
# ---------------------------------------------------------------------------


# Auto-skip if no nvcc; discover from canonical locations on the pod.
_nvcc = find_nvcc() or shutil.which("nvcc")
requires_nvcc = pytest.mark.skipif(
    _nvcc is None, reason="nvcc not available on PATH"
)


@requires_nvcc
class TestCompileAndLoad:
    def test_compile_relu_2x_3(self, tmp_path) -> None:
        expr = Relu(Add(Mul(Input("x"), Const(2.0)), Const(3.0)))
        k = compile_and_load(
            expr, kernel_name="test_relu_2x_3", nvcc_path=_nvcc
        )
        assert isinstance(k, LoadedKernel)
        assert k.so_path.is_file()
        assert k.src_path.is_file()
        assert k.kernel_name == "test_relu_2x_3"

    def test_kernel_and_launcher_symbols(self) -> None:
        expr = Relu(Input("x"))
        k = compile_and_load(
            expr, kernel_name="test_relu_only", nvcc_path=_nvcc
        )
        # ctypes.CDLL raises AttributeError on missing symbol, but the
        # attribute lookup is lazy; accessing them via getattr forces it.
        assert getattr(k.lib, "test_relu_only")
        assert getattr(k.lib, "launch_test_relu_only")

    def test_launcher_has_callable_signature(self) -> None:
        expr = Sigmoid(Input("x"))
        k = compile_and_load(
            expr, kernel_name="test_sigmoid", nvcc_path=_nvcc
        )
        launcher = k.launcher()
        assert callable(launcher)
        assert launcher.argtypes == [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]

    def test_cache_hit_skips_recompile(self, tmp_path) -> None:
        """Same expression + kernel_name should hit the .so cache."""
        expr = Tanh(Input("x"))
        k1 = compile_and_load(expr, kernel_name="test_tanh", nvcc_path=_nvcc)
        mtime = k1.so_path.stat().st_mtime_ns
        # Second invocation: should be a no-op build, same .so file.
        k2 = compile_and_load(expr, kernel_name="test_tanh", nvcc_path=_nvcc)
        assert k1.so_path == k2.so_path
        assert k2.so_path.stat().st_mtime_ns == mtime


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
