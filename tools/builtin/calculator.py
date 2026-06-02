"""内置计算器工具。"""

import ast
import operator
from typing import Any, Callable

from tools.base import Tool


class CalculatorTool(Tool):
    """安全计算简单数学表达式的工具。"""

    _BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "计算安全的数学表达式，支持加减乘除、括号、幂、取模和一元正负号"

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如：'(1 + 2) * 3'",
                },
            },
            "required": ["expression"],
        }

    def run(self, **kwargs: Any) -> str:
        expression = str(kwargs.get("expression", "")).strip()
        if not expression:
            return "错误：expression 不能为空"

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._evaluate(tree.body)
        except ZeroDivisionError:
            return "错误：除数不能为 0"
        except (SyntaxError, ValueError, TypeError, OverflowError) as e:
            return f"错误：表达式不合法 - {e}"

        return self._format_result(result)

    def _evaluate(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)

        if isinstance(node, ast.BinOp):
            operator_func = self._BINARY_OPERATORS.get(type(node.op))
            if operator_func is None:
                raise ValueError("不支持的运算符")
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            return operator_func(left, right)

        if isinstance(node, ast.UnaryOp):
            operator_func = self._UNARY_OPERATORS.get(type(node.op))
            if operator_func is None:
                raise ValueError("不支持的一元运算符")
            return operator_func(self._evaluate(node.operand))

        raise ValueError("只支持数字和基本数学运算")

    @staticmethod
    def _format_result(result: float) -> str:
        if result.is_integer():
            return str(int(result))
        return str(result)
