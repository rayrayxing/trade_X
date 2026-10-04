"""Safe, vectorised evaluation of strategy rule strings such as
``"close > ema_slow and pullback and engulf > 0"``.

Only names, numbers, comparisons, arithmetic, and/or/not and a short list of
helper functions are allowed. Nothing else in Python is reachable.
"""
from __future__ import annotations

import ast
from typing import Callable

import numpy as np
import pandas as pd


def _as_bool(x) -> pd.Series:
    return x.fillna(False).astype(bool) if isinstance(x, pd.Series) else x


def _cross_above(a, b):
    return (a > b) & (a.shift(1) <= (b.shift(1) if isinstance(b, pd.Series) else b))


def _cross_below(a, b):
    return (a < b) & (a.shift(1) >= (b.shift(1) if isinstance(b, pd.Series) else b))


FUNCS: dict[str, Callable] = {
    "shift": lambda x, n=1: x.shift(int(n)),
    "cross_above": _cross_above,
    "cross_below": _cross_below,
    "abs": lambda x: x.abs() if isinstance(x, pd.Series) else abs(x),
    "rolling_max": lambda x, n: x.rolling(int(n)).max(),
    "rolling_min": lambda x, n: x.rolling(int(n)).min(),
    "rolling_mean": lambda x, n: x.rolling(int(n)).mean(),
    "rising": lambda x, n=1: x > x.shift(int(n)),
    "falling": lambda x, n=1: x < x.shift(int(n)),
}

_CMP = {
    ast.Gt: lambda a, b: a > b, ast.Lt: lambda a, b: a < b, ast.GtE: lambda a, b: a >= b,
    ast.LtE: lambda a, b: a <= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}
_BIN = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
}


class ExprError(ValueError):
    pass


def names_in(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called


def evaluate(expr: str, env: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ExprError(f"unknown name {node.id!r} in rule {expr!r}")
            return env[node.id]
        if isinstance(node, ast.BoolOp):
            vals = [_as_bool(_truthy(ev(v))) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return ~_as_bool(_truthy(v))
            if isinstance(node.op, ast.USub):
                return -v
            raise ExprError(f"operator not allowed in {expr!r}")
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            out = None
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if type(op) not in _CMP:
                    raise ExprError(f"comparison not allowed in {expr!r}")
                r = _CMP[type(op)](left, right)
                out = r if out is None else (out & r)
                left = right
            return out
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BIN:
                raise ExprError(f"operator not allowed in {expr!r}")
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and not node.keywords:
            return FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ExprError(f"syntax not allowed in rule {expr!r}: {ast.dump(node)[:60]}")

    res = ev(tree)
    if not isinstance(res, pd.Series):
        res = pd.Series(bool(res), index=index)
    return _as_bool(_truthy(res)).reindex(index, fill_value=False)


def _truthy(v):
    """A bare numeric feature in a boolean position means 'non-zero and not NaN'."""
    if isinstance(v, pd.Series) and v.dtype != bool:
        return v.fillna(0).astype(float) != 0
    return v
