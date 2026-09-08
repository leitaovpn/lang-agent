"""内置无副作用演示工具。"""
import ast
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")


class CalculatorArgs(BaseModel):
    expression: str = Field(
        description="仅含数字、括号与 + - * / ** % // 的数学表达式，如 '(3+5)*7'"
    )


def calculator(expression: str) -> str:
    """安全计算数学表达式，返回字符串结果。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"无法计算表达式 {expression!r}: {exc}")
    return str(result)


class StringReverseArgs(BaseModel):
    text: str = Field(description="要反转的字符串")


def string_reverse(text: str) -> str:
    """反转字符串。"""
    return text[::-1]


class StringLenArgs(BaseModel):
    text: str = Field(description="要计算长度的字符串")


def string_len(text: str) -> int:
    """返回字符串长度。"""
    return len(text)
