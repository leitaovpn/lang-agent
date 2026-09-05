"""修复层测试：checkpoint 持久化历史必须满足 tool_call 不变式。

不变式：AIMessage 若含 tool_calls，其后必须为每条 tool_call 配一条 ToolMessage，
且 tool_call_id 在「两个 AIMessage 之间」不重合。
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lang_agent.core.repair import repair_state_for_checkpoint


def test_state_repair_fixes_segment_before_trailing_text_ai():
    # 坏段 [AI(dup 调用), TM(dup), TM(dup)] 后面跟了纯文本 AIMessage：
    # 持久化修复必须越过它，修到坏段本身
    messages = [
        HumanMessage(content="算两个"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup"},
                {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup"},
            ],
        ),
        ToolMessage(content="2", tool_call_id="dup", name="calculator"),
        ToolMessage(content="4", tool_call_id="dup", name="calculator"),
        AIMessage(content="分别是 2 和 4"),
    ]
    repaired = repair_state_for_checkpoint(messages)
    calling_ai = [m for m in repaired if isinstance(m, AIMessage) and m.tool_calls][0]
    ids = [c["id"] for c in calling_ai.tool_calls]
    assert len(ids) == len(set(ids)) == 2
    tool_ids = [m.tool_call_id for m in repaired if isinstance(m, ToolMessage)]
    assert sorted(tool_ids) == sorted(ids)
    # 尾部的纯文本 AIMessage 保留
    assert repaired[-1] == messages[-1]


def test_state_repair_missing_tool_message_synthesizes_error():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "c1"}]),
    ]
    repaired = repair_state_for_checkpoint(messages)
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert "未返回结果" in tool_msgs[0].content
    # 顺序：合成的 ToolMessage 紧跟 AIMessage 之后
    assert isinstance(repaired[-2], AIMessage)
    assert isinstance(repaired[-1], ToolMessage)


def test_state_repair_duplicate_tool_messages_dropped():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "c1"}]),
        ToolMessage(content="3", tool_call_id="c1", name="calculator"),
        ToolMessage(content="3 again", tool_call_id="c1", name="calculator"),
    ]
    repaired = repair_state_for_checkpoint(messages)
    tool_msgs = [m for m in repaired if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "3"


def test_state_repair_no_calling_ai_keeps_unchanged():
    messages = [HumanMessage(content="hi"), AIMessage(content="你好")]
    assert repair_state_for_checkpoint(messages) == messages
