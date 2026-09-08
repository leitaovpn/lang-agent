"""工具注册表：内置演示工具 + 扩展口（不做 tool 鉴权）。"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_community.agent_toolkits.file_management import FileManagementToolkit
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from lang_agent.core import tools as builtin_tools


@dataclass(slots=True)
class ToolSpec:
    """一个工具的完整描述。"""

    name: str
    description: str
    fn: Callable[..., Any]
    args_schema: type[BaseModel] | None = None


_TOOLS: dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    """注册一个工具（重复注册同名工具会覆盖）。"""
    _TOOLS[spec.name] = spec


def unregister_tool(name: str) -> None:
    _TOOLS.pop(name, None)


def get_tool(name: str) -> ToolSpec:
    try:
        return _TOOLS[name]
    except KeyError:
        raise ValueError(f"未知工具: {name!r}，已注册: {sorted(_TOOLS)}")


def instantiate_tools(names: list[str] | None = None) -> list[BaseTool]:
    """把注册表转为 langchain BaseTool 列表，供 bind_tools / ToolNode 使用。"""
    specs = [_TOOLS[n] for n in names] if names else list(_TOOLS.values())
    return [
        StructuredTool.from_function(
            func=spec.fn,
            name=spec.name,
            description=spec.description,
            args_schema=spec.args_schema,
        )
        for spec in specs
    ] + FileManagementToolkit().get_tools() + load_tools(["terminal", "ddg-search", "wikipedia", "arxiv", "pubmed"], allow_dangerous_tools=True) # 内置文件管理工具


# ---- 内置演示工具 ----
register_tool(
    ToolSpec(
        name="calculator",
        description="安全计算数学表达式（仅支持数字、括号与 + - * / ** % // 运算），如 '(3+5)*7'",
        fn=builtin_tools.calculator,
        args_schema=builtin_tools.CalculatorArgs,
    )
)
register_tool(
    ToolSpec(
        name="string_reverse",
        description="反转字符串",
        fn=builtin_tools.string_reverse,
        args_schema=builtin_tools.StringReverseArgs,
    )
)
register_tool(
    ToolSpec(
        name="string_len",
        description="返回字符串长度",
        fn=builtin_tools.string_len,
        args_schema=builtin_tools.StringLenArgs,
    )
)

