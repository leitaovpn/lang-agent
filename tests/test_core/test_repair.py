"""repair_messages_for_llm：发给 LLM 的历史不变式修复测试。

不变式：AIMessage 若含 tool_calls，其后必须为每条 tool_call 配一条 ToolMessage，
且 tool_call_id 全局不重复。
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lang_agent.core.repair import repair_messages_for_llm


def test_valid_conversation_untouched():
    messages = [HumanMessage(content="hi"), AIMessage(content="你好")]
    assert repair_messages_for_llm(messages) == messages


def test_missing_tool_message_synthesizes_error():
    messages = [
        HumanMessage(content="算一下"),
        AIMessage(
            content="",
            tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}],
        ),
    ]
    repaired = repair_messages_for_llm(messages)
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert "未返回结果" in tool_msgs[0].content
    # 顺序：合成的 ToolMessage 紧跟 AIMessage 之后
    assert isinstance(repaired[-2], AIMessage)
    assert isinstance(repaired[-1], ToolMessage)


def test_duplicate_tool_call_ids_are_renamed():
    messages = [
        HumanMessage(content="算两个"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup"},
                {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup"},
            ],
        ),
    ]
    repaired = repair_messages_for_llm(messages)
    ai = [m for m in repaired if isinstance(m, AIMessage)][0]
    ids = [c["id"] for c in ai.tool_calls]
    assert len(ids) == 2 and len(set(ids)) == 2
    # 每条调用都补了 ToolMessage，且 id 与改名后一一对应
    tool_ids = [m.tool_call_id for m in repaired if isinstance(m, ToolMessage)]
    assert sorted(tool_ids) == sorted(ids)


def test_duplicate_tool_messages_are_dropped():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "c1"}]),
        ToolMessage(content="3", tool_call_id="c1", name="calculator"),
        ToolMessage(content="3 again", tool_call_id="c1", name="calculator"),
    ]
    repaired = repair_messages_for_llm(messages)
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "3"


def test_orphan_tool_message_dropped():
    messages = [ToolMessage(content="孤儿", tool_call_id="ghost", name="x")]
    assert repair_messages_for_llm(messages) == []


def test_invalid_tool_calls_synthesize_error_feedback():
    messages = [
        HumanMessage(content="算一下"),
        AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[
                {"name": "calculator", "args": "{bad", "id": None, "error": "JSON 解析失败"}
            ],
        ),
    ]
    repaired = repair_messages_for_llm(messages)
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "格式错误" in tool_msgs[0].content
    assert "JSON 解析失败" in tool_msgs[0].content


def test_invalid_feedback_not_duplicated_on_second_repair():
    messages = [
        HumanMessage(content="算一下"),
        AIMessage(
            content="",
            tool_calls=[],
            invalid_tool_calls=[
                {"name": "calculator", "args": "{bad", "id": None, "error": "JSON 解析失败"}
            ],
        ),
    ]
    once = repair_messages_for_llm(messages)
    twice = repair_messages_for_llm(once)
    assert twice == once


def test_user_message_inserts_error_before_it():
    # AIMessage 带 tool_call 但结果还没回来就来了新用户消息 → 先补错误结果再放用户消息
    messages = [
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "c1"}]),
        HumanMessage(content="打断一下"),
    ]
    repaired = repair_messages_for_llm(messages)
    assert isinstance(repaired[0], AIMessage)
    assert isinstance(repaired[1], ToolMessage)
    assert repaired[1].tool_call_id == "c1"
    assert repaired[2] == messages[1]


def test_previous_history_is_trusted_inductively():
    # 归纳不变式：LLM 只会基于已修复的历史回复新 AIMessage，因此最后一条
    # AIMessage 之前的结构必然正确——修复只处理尾部，前缀原样保留
    messages = [
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "old"}]),
        HumanMessage(content="继续"),
        AIMessage(content="好的"),
    ]
    assert repair_messages_for_llm(messages) == messages


def test_earlier_call_ids_ignored_for_dedup():
    # 去重范围只限「两个 AIMessage 之间」：最后一条 AIMessage 之前的
    # tool_call_id 不参与处理，尾部与历史重名不改
    messages = [
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "old"}]),
        ToolMessage(content="3", tool_call_id="old", name="calculator"),
        AIMessage(content="", tool_calls=[{"name": "string_len", "args": {"text": "abc"}, "id": "old"}]),
    ]
    repaired = repair_messages_for_llm(messages)
    tail_index = [i for i, m in enumerate(repaired) if isinstance(m, AIMessage)][-1]
    tail_ai = repaired[tail_index]
    assert tail_ai.tool_calls[0]["id"] == "old"
    # 但尾部内部不重合、ToolMessage 一一对应（只统计尾部段，前缀同 id 的除外）
    tail_ids = [c["id"] for c in tail_ai.tool_calls]
    assert len(tail_ids) == len(set(tail_ids))
    tail_tools = [
        m for m in repaired[tail_index:] if isinstance(m, ToolMessage) and m.tool_call_id in tail_ids
    ]
    assert len(tail_tools) == len(tail_ids)


def test_no_aimessage_keeps_non_tool_messages():
    # 首轮：只有用户消息 → 原样返回
    messages = [HumanMessage(content="hi")]
    assert repair_messages_for_llm(messages) == messages
