"""CalculatorTool tests."""

from tools.builtin.calculator import CalculatorTool


def test_calculator_tool_evaluates_basic_expression() -> None:
    tool = CalculatorTool()

    result = tool.run(expression="(1 + 2) * 3")

    assert result == "9"


def test_calculator_tool_supports_float_result() -> None:
    tool = CalculatorTool()

    result = tool.run(expression="7 / 2")

    assert result == "3.5"


def test_calculator_tool_rejects_unsafe_expression() -> None:
    tool = CalculatorTool()

    result = tool.run(expression="__import__('os').system('echo unsafe')")

    assert result.startswith("错误：表达式不合法")
