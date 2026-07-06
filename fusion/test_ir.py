"""
fusion/test_ir.py - IR correctness tests (no nvcc / GPU required).

These run on a plain Python interpreter; they verify that the AST-shaped
IR nodes lower to the correct C++ expression strings.
"""

from __future__ import annotations

import pytest

from fusion.ir import Input, Const, Mul, Add, Relu, Sigmoid, Tanh, Expr


class TestLeaves:
    def test_input_default(self) -> None:
        assert Input().codegen() == "x"

    def test_input_named(self) -> None:
        assert Input("a").codegen() == "a"

    def test_const_float(self) -> None:
        assert Const(2.0).codegen() == "2.000000f"

    def test_const_int_promotes_to_float(self) -> None:
        # ints widen to float so the literal has the `f` suffix.
        assert Const(3).codegen() == "3.000000f"

    def test_const_negative(self) -> None:
        assert Const(-0.5).codegen() == "-0.500000f"


class TestBinary:
    def test_mul(self) -> None:
        assert Mul(Input("x"), Const(2.0)).codegen() == "(x * 2.000000f)"

    def test_add(self) -> None:
        assert Add(Input("x"), Const(3.0)).codegen() == "(x + 3.000000f)"

    def test_mul_parenthesises_nested(self) -> None:
        e = Mul(Add(Input("x"), Const(1.0)), Const(2.0))
        assert e.codegen() == "((x + 1.000000f) * 2.000000f)"


class TestUnary:
    def test_relu(self) -> None:
        assert Relu(Input("x")).codegen() == "fmaxf(x, 0.0f)"

    def test_sigmoid(self) -> None:
        assert Sigmoid(Input("x")).codegen() == "(1.0f / (1.0f + expf(-(x))))"

    def test_tanh(self) -> None:
        assert Tanh(Input("x")).codegen() == "tanhf(x)"


class TestComposition:
    def test_relu_x2_plus_3(self) -> None:
        e = Relu(Add(Mul(Input("x"), Const(2.0)), Const(3.0)))
        # Extra parens around the inner Add are valid C++ but loud to humans.
        assert e.codegen() == "fmaxf(((x * 2.000000f) + 3.000000f), 0.0f)"

    def test_relu_neg_x(self) -> None:
        # y = relu(-x)  --> fmaxf(-x, 0)
        e = Relu(Mul(Input("x"), Const(-1.0)))
        assert e.codegen() == "fmaxf((x * -1.000000f), 0.0f)"

    def test_nested_three_levels(self) -> None:
        e = Sigmoid(Add(Relu(Input("x")), Const(1.0)))
        expected = "(1.0f / (1.0f + expf(-((fmaxf(x, 0.0f) + 1.000000f)))))"
        assert e.codegen() == expected


class TestStructural:
    def test_all_nodes_are_exprs(self) -> None:
        nodes = [
            Input("x"),
            Const(1.0),
            Mul(Input("x"), Const(2.0)),
            Add(Input("x"), Const(3.0)),
            Relu(Input("x")),
            Sigmoid(Input("x")),
            Tanh(Input("x")),
        ]
        for n in nodes:
            assert isinstance(n, Expr)

    def test_codegen_is_pure(self) -> None:
        # Calling codegen() twice yields the same string (no hidden state).
        e = Relu(Add(Mul(Input("x"), Const(2.0)), Const(3.0)))
        assert e.codegen() == e.codegen()


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
