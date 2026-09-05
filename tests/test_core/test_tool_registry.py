"""工具注册表测试。"""
import pytest
from langchain_core.tools import BaseTool

from lang_agent.core.tool_registry import (
    get_tool,
    instantiate_tools,
    register_tool,
    unregister_tool,
)


def test_default_tools_registered():
    names = {t.name for t in instantiate_tools()}
    assert {"calculator", "string_reverse", "string_len"} <= names


def test_get_tool_returns_spec():
    spec = get_tool("calculator")
    assert spec.name == "calculator"
    assert spec.description


def test_get_tool_unknown_raises():
    with pytest.raises(ValueError):
        get_tool("no_such_tool")


def test_instantiate_tools_returns_base_tools():
    tools = instantiate_tools()
    assert tools
    assert all(isinstance(t, BaseTool) for t in tools)


def test_register_tool_adds_new_tool():
    from lang_agent.core.tool_registry import ToolSpec

    spec = ToolSpec(name="test_echo", description="测试用 echo 工具", fn=lambda s: s, args_schema=None)
    register_tool(spec)
    try:
        assert "test_echo" in {t.name for t in instantiate_tools()}
        assert get_tool("test_echo").fn("hi") == "hi"
    finally:
        unregister_tool("test_echo")
