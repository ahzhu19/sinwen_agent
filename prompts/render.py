"""Prompt 模板渲染。"""


def render_prompt(template: str, **variables: object) -> str:
    """渲染 prompt 模板，并把缺失变量转换成更清晰的错误。"""
    try:
        return template.format(**variables)
    except KeyError as e:
        missing_key = e.args[0]
        raise ValueError(f"Prompt 缺少变量: {missing_key}") from e
