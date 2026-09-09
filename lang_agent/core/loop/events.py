"""agent 事件协议：core 层对外的统一事件形态与分类纯函数。

事件类型：
- thinking_token  模型逐 token 思考内容（reasoning_content，来自 agent chunk 的 additional_kwargs）
- llm_token   agent 逐 token 文本（来自 messages 流式通道）
- tool_call   模型发起一次工具调用（来自 updates 通道的完整 AIMessage）
- tool_result 工具执行结果（来自 updates 通道的 ToolMessage）
- done        循环正常结束，携带最终文本与工具调用汇总
- error       循环异常终止，携带错误信息
"""
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

EVENT_THINKING_TOKEN = "thinking_token"
EVENT_LLM_TOKEN = "llm_token"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_DONE = "done"
EVENT_ERROR = "error"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        """序列化为 SSE 帧：event: <type>\\ndata: <json>\\n\\n"""
        return (
            f"event: {self.type}\n"
            f"data: {json.dumps(self.data, ensure_ascii=False)}\n\n"
        )


@dataclass(slots=True)
class ConversationResult:
    """invoke() 的同步结果。messages 与 tool_calls 都只含本轮（最后一条 HumanMessage 起）产生的内容。"""

    thread_id: str
    final_text: str
    messages: list[BaseMessage] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def classify_message_chunk(
    chunk: BaseMessage, metadata: dict[str, Any]
) -> AgentEvent | None:
    """messages 通道的 chunk 分类：agent 节点的 thinking → thinking_token，文本 → llm_token。

    thinking 增量位于 chunk.additional_kwargs["reasoning_content"]（deepseek 系
    模型流式行为）；后续内容（content）走 llm_token。
    """
    if metadata.get("langgraph_node") != "agent":
        return None
    if not isinstance(chunk, AIMessageChunk):
        return None
    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
    if isinstance(reasoning, str) and reasoning:
        return AgentEvent(EVENT_THINKING_TOKEN, {"text": reasoning})
    if isinstance(chunk.content, str) and chunk.content:
        return AgentEvent(EVENT_LLM_TOKEN, {"text": chunk.content})
    return None


def classify_node_update(node: str, delta: dict[str, Any]) -> list[AgentEvent]:
    """updates 通道的节点增量分类：agent → tool_call；tools → tool_result。"""
    events: list[AgentEvent] = []
    for message in delta.get("messages", []):
        if node == "agent" and isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                events.append(
                    AgentEvent(
                        EVENT_TOOL_CALL,
                        {
                            "id": tool_call.get("id"),
                            "name": tool_call.get("name"),
                            "arguments": tool_call.get("args", {}),
                        },
                    )
                )
        elif node == "tools" and isinstance(message, ToolMessage):
            events.append(
                AgentEvent(
                    EVENT_TOOL_RESULT,
                    {
                        "tool_call_id": message.tool_call_id,
                        "name": message.name,
                        "content": message.content,
                    },
                )
            )
    return events


def messages_after_last_human(messages: list[BaseMessage]) -> list[BaseMessage]:
    """截取最近一条 HumanMessage 起的本轮消息（含其后的 AIMessage/ToolMessage）。

    每次 invoke 都会追加一条新的 HumanMessage，因此最后一条 HumanMessage
    就是本轮循环的起点；之前的轮次不参与本轮汇总。
    """
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return messages[idx:]
    return messages


def collect_tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """汇总给定消息范围内 AIMessage 发起过的工具调用（调用方负责传本轮切片）。"""
    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                tool_calls.append(
                    {
                        "id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "arguments": tool_call.get("args", {}),
                    }
                )
    return tool_calls
