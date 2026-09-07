"""context 压缩纯函数测试：切点规则、摘要渲染、工具输出截断。"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lang_agent.core.compress import (
    estimate_tokens,
    find_round_start,
    render_messages_for_summary,
    truncate_tool_outputs,
)


def _round1():
    return [
        HumanMessage(content="问题一"),
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "1+2"}, "id": "c1"}]),
        ToolMessage(content="3", tool_call_id="c1", name="calculator"),
        AIMessage(content="答案是 3"),
    ]


def test_estimate_tokens_falls_back_to_chars_when_counter_unavailable():
    # FakeChatModel 的精确计数依赖 transformers（未装）→ 抛 ImportError → 字符/4 兜底
    from tests.conftest import FakeChatModel

    llm = FakeChatModel(responses=[AIMessage(content="x")])
    messages = [HumanMessage(content="你好呀呀"), AIMessage(content="你好")]
    assert estimate_tokens(messages, llm) == estimate_tokens(messages, None) > 0


def test_find_round_start_moves_to_human():
    messages = _round1() + [HumanMessage(content="问题二"), AIMessage(content="回答二")]
    assert find_round_start(messages, len(messages) - 1) == 4


def test_find_round_start_never_splits_tool_call_segment():
    # 切点候选落在轮中间（tool_call 段内）→ 回退到轮起点
    messages = _round1() + [HumanMessage(content="问题二"), AIMessage(content="回答二")]
    assert find_round_start(messages, 2) == 0
    assert find_round_start(messages, 5) == 4


def test_find_round_start_no_human_returns_zero():
    messages = [AIMessage(content="hi")]
    assert find_round_start(messages, 0) == 0


def test_render_messages_for_summary():
    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好呀"),
        ToolMessage(content="3", tool_call_id="c1", name="calculator"),
    ]
    text = render_messages_for_summary(messages)
    assert "user: 你好" in text
    assert "assistant: 你好呀" in text
    assert "tool(calculator): 3" in text


def test_truncate_tool_outputs_truncates_long_content():
    messages = [ToolMessage(content="x" * 100, tool_call_id="c1", name="t")]
    truncated = truncate_tool_outputs(messages, max_chars=10)
    assert len(truncated[0].content) <= 10 + len("…(已截断)")
    assert "已截断" in truncated[0].content


def test_truncate_tool_outputs_keeps_short_and_structure():
    messages = [
        HumanMessage(content="hi"),
        ToolMessage(content="短内容", tool_call_id="c1", name="t"),
        AIMessage(content="好"),
    ]
    out = truncate_tool_outputs(messages, max_chars=100)
    assert out[1].content == "短内容"
    assert isinstance(out[0], HumanMessage)
    assert isinstance(out[2], AIMessage)
