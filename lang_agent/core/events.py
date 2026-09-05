"""agent 事件协议：core 层对外的统一事件形态与分类纯函数。

事件类型：
- llm_token   agent 逐 token 文本（来自 messages 流式通道）
- tool_call   模型发起一次工具调用（来自 updates 通道的完整 AIMessage）
- tool_result 工具执行结果（来自 updates 通道的 ToolMessage）
- done        循环正常结束，携带最终文本与工具调用汇总
- error       循环异常终止，携带错误信息
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage

EVENT_LLM_TOKEN = "llm_token"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_DONE = "done"
EVENT_ERROR = "error"


@dataclass(frozen=True)
class AgentEvent:
    type: str
    data: Dict[str, Any]

    def to_sse(self) -> str:
        """序列化为 SSE 帧：event: <type>\\ndata: <json>\\n\\n"""
        return "event: %s\ndata: %s\n\n" % (
            self.type,
            json.dumps(self.data, ensure_ascii=False),
        )


@dataclass
class ConversationResult:
    """invoke() 的同步结果。"""

    thread_id: str
    final_text: str
    messages: List[BaseMessage] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


def classify_message_chunk(
    chunk: BaseMessage, metadata: Dict[str, Any]
) -> Optional[AgentEvent]:
    """messages 通道的 chunk 分类：agent 节点的文本 token → llm_token。"""
    if metadata.get("langgraph_node") != "agent":
        return None
    if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and chunk.content:
        return AgentEvent(EVENT_LLM_TOKEN, {"text": chunk.content})
    return None


def classify_node_update(node: str, delta: Dict[str, Any]) -> List[AgentEvent]:
    """updates 通道的节点增量分类：agent → tool_call；tools → tool_result。"""
    events: List[AgentEvent] = []
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


def collect_tool_calls(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """从消息历史中汇总所有 AIMessage 发起过的工具调用。"""
    tool_calls: List[Dict[str, Any]] = []
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
