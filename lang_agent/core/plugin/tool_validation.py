"""工具 wrapper 的输出约束。"""

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from .types import PluginError


def validate_tool_result(result, call_id):
    if isinstance(result, ToolMessage):
        if result.tool_call_id != call_id:
            raise PluginError("工具结果的 tool_call_id 不匹配")
        return
    if isinstance(result, Command):
        if result.goto or result.graph or not isinstance(result.update, dict):
            raise PluginError("工具 Command 不允许跳转")
        if result.update.keys() - {"messages"}:
            raise PluginError("工具 Command 包含受保护字段")
        messages = result.update.get("messages", [])
        if not messages or any(
            not isinstance(m, ToolMessage) or m.tool_call_id != call_id
            for m in messages
        ):
            raise PluginError("工具 Command 必须包含对应 ToolMessage")
        return
    raise PluginError("工具 wrapper 必须返回 ToolMessage 或受限 Command")
