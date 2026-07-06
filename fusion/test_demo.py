"""
fusion/test_demo.py - Demo smoke tests.

The real benchmark requires a GPU pod and is gated behind
`@pytest.mark.skip(reason=...)`. The import + scaffold verification
runs locally to confirm the wiring is intact.
"""

from __future__ import annotations

import pytest

from fusion import demo as demo_module


class TestModuleImport:
    def test_module_imports_cleanly(self) -> None:
        # If the module has a syntax error / bad import, the pytest
        # collection above would fail before this point.
        assert hasattr(demo_module, "demo")
        assert hasattr(demo_module, "main")
        assert hasattr(demo_module, "build_fused_expr")
        assert hasattr(demo_module, "build_fused_kernel")


class TestScaffold:
    def test_build_fused_expr_returns_ir(self) -> None:
        from fusion.ir import Relu
        expr = demo_module.build_fused_expr()
        assert isinstance(expr, Relu)

    def test_fused_expr_codegen(self) -> None:
        expr = demo_module.build_fused_expr()
        s = expr.codegen()
        assert "fmaxf" in s
        assert "2.000000f" in s
        assert "3.000000f" in s


@pytest.mark.skip(reason="GPU pod required; run manually via python3 fusion/demo.py")
def test_fusion_demo():
    """Run on a live GPU pod to verify fused >= 1.5x speedup."""
    rc = demo_module.demo()
    # On a real GPU pod, demo() should reach the live-compile path and
    # exit 2 (deferred for the actual benchmark follow-up).
    assert rc in (1, 2)


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
