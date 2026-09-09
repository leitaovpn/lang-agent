"""事件分类纯函数与 SSE 序列化测试。"""
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from lang_agent.core.loop.events import (
    AgentEvent,
    classify_message_chunk,
    classify_node_update,
    messages_after_last_human,
)


def test_messages_after_last_human_slices_current_round():
    messages = [
        HumanMessage(content="第一问"),
        AIMessage(content="第一答"),
        HumanMessage(content="第二问"),
        AIMessage(content="第二答"),
    ]
    assert messages_after_last_human(messages) == messages[2:]


def test_messages_after_last_human_without_human_returns_all():
    messages = [AIMessage(content="答")]
    assert messages_after_last_human(messages) == messages


def test_token_from_agent_chunk():
    chunk = AIMessageChunk(content="你好")
    event = classify_message_chunk(chunk, {"langgraph_node": "agent"})
    assert event == AgentEvent("llm_token", {"text": "你好"})


def test_thinking_chunk_emits_thinking_token():
    chunk = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "想"})
    event = classify_message_chunk(chunk, {"langgraph_node": "agent"})
    assert event == AgentEvent("thinking_token", {"text": "想"})


def test_thinking_then_content_chunks_classify_separately():
    chunk = AIMessageChunk(content="", additional_kwargs={"reasoning_content": "想"})
    assert classify_message_chunk(chunk, {"langgraph_node": "agent"}).type == "thinking_token"
    chunk2 = AIMessageChunk(content="答")
    assert classify_message_chunk(chunk2, {"langgraph_node": "agent"}).type == "llm_token"


def test_chunk_from_tools_node_ignored():
    chunk = AIMessageChunk(content="56")
    assert classify_message_chunk(chunk, {"langgraph_node": "tools"}) is None


def test_empty_chunk_ignored():
    chunk = AIMessageChunk(content="")
    assert classify_message_chunk(chunk, {"langgraph_node": "agent"}) is None


def test_agent_update_with_tool_calls_emits_tool_call():
    message = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "call_1"}],
    )
    events = classify_node_update("agent", {"messages": [message]})
    assert events == [AgentEvent("tool_call", {"id": "call_1", "name": "calculator", "arguments": {"expression": "1+1"}})]


def test_agent_update_plain_text_emits_nothing_extra():
    # 纯文本答案只记录为最终文本（由 AgentLoop 处理），分类器不发重复 token 事件
    message = AIMessage(content="答案是 42")
    assert classify_node_update("agent", {"messages": [message]}) == []


def test_tools_update_emits_tool_result():
    message = ToolMessage(content="56", tool_call_id="call_1", name="calculator")
    events = classify_node_update("tools", {"messages": [message]})
    assert events == [AgentEvent("tool_result", {"tool_call_id": "call_1", "name": "calculator", "content": "56"})]


def test_sse_serialization():
    event = AgentEvent("llm_token", {"text": "你好"})
    sse = event.to_sse()
    assert sse == 'event: llm_token\ndata: {"text": "你好"}\n\n'
