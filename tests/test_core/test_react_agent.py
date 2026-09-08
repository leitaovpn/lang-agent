"""AgentLoop（ReAct graph 薄封装）行为测试，graph 直驱，全部用 FakeChatModel。

会话入口（repair/compress/retry/事件分类）的测试见 tests/test_agent/test_session.py。
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from lang_agent.core import (
    AgentContext,
    AgentLoop,
    AgentLoopConfig,
    build_checkpointer,
)
from lang_agent.core.tool_registry import instantiate_tools
from tests.conftest import FakeChatModel

TOOL_CALL = {
    "name": "calculator",
    "args": {"expression": "(3+5)*7"},
    "id": "call_1",
    "type": "tool_call",
}


def make_loop(script, *, checkpointer=None, config=None, agent_node=None, tools_node=None):
    llm = FakeChatModel(responses=list(script))
    loop = AgentLoop(
        config=config,
        checkpointer=checkpointer,
        agent_node=agent_node,
        tools_node=tools_node,
    )
    return loop, llm


def make_ctx(llm, tools=None, summarizer_llm=None) -> AgentContext:
    return AgentContext(
        llm=llm,
        tools=tools if tools is not None else instantiate_tools(),
        summarizer_llm=summarizer_llm,
    )


async def run_graph(loop, query, *, ctx, thread_id="t1", system=None) -> dict:
    """组装初始 state 与 config，直驱 graph.ainvoke（与 session 层的组装等价）。"""
    input_ = {"messages": [HumanMessage(content=query)], "raw_input": query}
    if system:
        input_["system"] = system
    return await loop.graph.ainvoke(
        input_, {"configurable": {"thread_id": thread_id}}, context=ctx
    )


async def test_system_message_prepended():
    loop, llm = make_loop([AIMessage(content="好")])
    await run_graph(loop, "问", ctx=make_ctx(llm), system="你是计算助手")
    assert llm.seen_messages[0][0] == SystemMessage(content="你是计算助手")


async def test_memory_second_round_sees_history():
    loop, llm = make_loop(
        [AIMessage(content="第一次回答"), AIMessage(content="第二次回答")],
        checkpointer=InMemorySaver(),
    )
    ctx = make_ctx(llm)
    await run_graph(loop, "第一问", ctx=ctx)
    await run_graph(loop, "第二问", ctx=ctx)
    seen = llm.seen_messages[1]
    assert len(seen) >= 3
    assert isinstance(seen[0], HumanMessage) and seen[0].content == "第一问"
    assert isinstance(seen[-1], HumanMessage) and seen[-1].content == "第二问"


async def test_different_thread_ids_isolated():
    loop, llm = make_loop(
        [AIMessage(content="回答1"), AIMessage(content="回答2")],
        checkpointer=InMemorySaver(),
    )
    ctx = make_ctx(llm)
    await run_graph(loop, "问", ctx=ctx, thread_id="t1")
    await run_graph(loop, "问", ctx=ctx, thread_id="t2")
    for seen in llm.seen_messages:
        assert len(seen) == 1
        assert isinstance(seen[0], HumanMessage) and seen[0].content == "问"


async def test_loop_retries_on_invalid_tool_calls():
    # 第一轮返回解析失败的 tool_call → 循环应附错误反馈并重试，最终给出回答
    invalid = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "calculator", "args": "{bad", "id": None, "error": "JSON 解析失败"}
        ],
    )
    loop, llm = make_loop([invalid, AIMessage(content="抱歉，我重新算：答案是 42")])
    state = await run_graph(loop, "算一下 1+1", ctx=make_ctx(llm))
    assert state["messages"][-1].content == "抱歉，我重新算：答案是 42"
    # 第二轮 LLM 看到的历史里包含错误反馈 ToolMessage
    seen = llm.seen_messages[1]
    assert any(isinstance(m, ToolMessage) and "格式错误" in m.content for m in seen)


async def test_loop_completes_with_duplicate_tool_call_ids():
    # 模型返回重复 id 的 tool_call：工具照常执行、循环照常完成；
    # 历史修复由下次调用的入口 checkpoint 修复兜底（见 test_session.py）
    calls = [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup", "type": "tool_call"},
    ]
    loop, llm = make_loop([AIMessage(content="", tool_calls=calls), AIMessage(content="分别是 2 和 4")])
    state = await run_graph(loop, "算两个", ctx=make_ctx(llm))
    assert state["messages"][-1].content == "分别是 2 和 4"
    tool_contents = sorted(
        m.content for m in state["messages"] if isinstance(m, ToolMessage)
    )
    assert tool_contents == ["2", "4"]
    # 不做 LLM 拷贝修复：第二轮模型看到的历史保持原样（重复 id 原样传入）
    seen = llm.seen_messages[1]
    ai = next(m for m in seen if isinstance(m, AIMessage) and m.tool_calls)
    assert [c["id"] for c in ai.tool_calls] == ["dup", "dup"]


async def test_tool_execution_error_feeds_error_to_llm():
    # 工具执行异常（除零）：ToolNode 捕获后以错误 ToolMessage 投喂 LLM，循环继续
    calls = [
        {"name": "calculator", "args": {"expression": "1/0"}, "id": "c1", "type": "tool_call"}
    ]
    loop, llm = make_loop([AIMessage(content="", tool_calls=calls), AIMessage(content="除零失败，换个算法")])
    state = await run_graph(loop, "1 除以 0", ctx=make_ctx(llm))
    assert state["messages"][-1].content == "除零失败，换个算法"
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert "无法计算" in tool_msgs[0].content or "division by zero" in tool_msgs[0].content
    assert tool_msgs[0].status == "error"


async def test_unknown_tool_feeds_error_to_llm():
    # LLM 调用 ToolNode 范围外的工具：返回含报错信息的 ToolMessage，循环继续
    calls = [
        {"name": "no_such_tool", "args": {"x": 1}, "id": "c1", "type": "tool_call"}
    ]
    loop, llm = make_loop([AIMessage(content="", tool_calls=calls), AIMessage(content="没有这个工具，我直接回答")])
    state = await run_graph(loop, "用不存在的工具", ctx=make_ctx(llm))
    assert state["messages"][-1].content == "没有这个工具，我直接回答"
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert "not a valid tool" in tool_msgs[0].content


async def test_tool_output_truncation_in_send_view_only():
    long_text = "长" * 300
    loop, llm = make_loop(
        [
            AIMessage(content="", tool_calls=[{"name": "string_reverse", "args": {"text": long_text}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="已反转"),
        ],
        config=AgentLoopConfig(compress_tool_output_max_chars=10),
        checkpointer=InMemorySaver(),
    )
    state = await run_graph(loop, "反转", ctx=make_ctx(llm))
    assert state["messages"][-1].content == "已反转"
    # 发送视图：LLM 第二轮看到的工具输出被截断
    seen = llm.seen_messages[1]
    seen_tool = next(m for m in seen if isinstance(m, ToolMessage))
    assert "已截断" in seen_tool.content
    # checkpoint：完整内容保留
    ckpt = await loop.graph.aget_state({"configurable": {"thread_id": "t1"}})
    state_tool = next(m for m in ckpt.values["messages"] if isinstance(m, ToolMessage))
    assert len(state_tool.content) == len(long_text)


async def test_sqlite_checkpointer_persists_across_instances(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = str(tmp_path / "ck.sqlite")
    config = AgentLoopConfig(checkpointer_kind="sqlite", db_path=db_path)

    # async with 保证连接关闭，否则 aiosqlite 连接会让测试会话挂起
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver1:
        loop1, llm1 = make_loop([AIMessage(content="第一次回答")], config=config, checkpointer=saver1)
        await run_graph(loop1, "第一问", ctx=make_ctx(llm1))

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver2:
        loop2, llm2 = make_loop([AIMessage(content="第二次回答")], config=config, checkpointer=saver2)
        await run_graph(loop2, "第二问", ctx=make_ctx(llm2))
        seen = llm2.seen_messages[0]
        assert isinstance(seen[0], HumanMessage) and seen[0].content == "第一问"
        assert isinstance(seen[-1], HumanMessage) and seen[-1].content == "第二问"


async def test_build_checkpointer_expands_tilde(tmp_path, monkeypatch):
    # db_path 的 ~ 必须展开到 HOME：不展开会在工作目录下建出字面量 ~ 目录
    monkeypatch.setenv("HOME", str(tmp_path))
    config = AgentLoopConfig(checkpointer_kind="sqlite", db_path="~/ck.sqlite")
    saver = await build_checkpointer(config)
    try:
        assert (tmp_path / "ck.sqlite").exists()
    finally:
        await saver.conn.close()


async def test_build_checkpointer_creates_missing_parent_dir(tmp_path):
    db_path = str(tmp_path / "nested" / "dir" / "ck.sqlite")
    config = AgentLoopConfig(checkpointer_kind="sqlite", db_path=db_path)
    saver = await build_checkpointer(config)
    try:
        assert (tmp_path / "nested" / "dir").is_dir()
    finally:
        await saver.conn.close()


async def test_default_checkpointer_is_memory():
    loop, _ = make_loop([AIMessage(content="好")])
    assert loop.graph.checkpointer is not None


# ---- 新契约：节点注入 / context 必选 / 同形透传 ----


async def test_injected_agent_node_is_used():
    async def fake_agent_node(state, config, runtime):
        return {"messages": [AIMessage(content="注入节点回答")]}

    loop = AgentLoop(checkpointer=InMemorySaver(), agent_node=fake_agent_node)
    ctx = AgentContext(llm=FakeChatModel(responses=[]), tools=[])
    state = await loop.graph.ainvoke(
        {"messages": [HumanMessage(content="问")]},
        {"configurable": {"thread_id": "t1"}},
        context=ctx,
    )
    assert state["messages"][-1].content == "注入节点回答"


async def test_injected_tools_node_is_used():
    async def fake_tools_node(state, config, runtime):
        return {"messages": [ToolMessage(content="fake 工具结果", tool_call_id="c1", name="calculator")]}

    loop = AgentLoop(checkpointer=InMemorySaver(), tools_node=fake_tools_node)
    llm = FakeChatModel(responses=[AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="工具已执行")])
    state = await run_graph(loop, "算一下", ctx=make_ctx(llm))
    assert state["messages"][-1].content == "工具已执行"
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and tool_msgs[0].content == "fake 工具结果"


async def test_invoke_requires_context():
    loop, _ = make_loop([AIMessage(content="好")])
    with pytest.raises(ValueError, match="context"):
        await loop.invoke({"messages": []}, context=None)


def test_stream_requires_context():
    # stream 是普通函数：context 缺失在调用点同步抛错
    loop, _ = make_loop([AIMessage(content="好")])
    with pytest.raises(ValueError, match="context"):
        loop.stream({"messages": []}, context=None)


async def test_stream_passthrough_yields_raw_mode_payload_tuples():
    # 同形透传契约：stream 产出原始 (mode, payload) 元组，不做事件分类
    loop, llm = make_loop([AIMessage(content="结果是 56")])
    items = [
        item
        async for item in loop.stream(
            {"messages": [HumanMessage(content="计算")]},
            {"configurable": {"thread_id": "t1"}},
            context=make_ctx(llm),
            stream_mode=["messages", "updates"],
        )
    ]
    assert items
    assert all(isinstance(item, tuple) and len(item) == 2 for item in items)
    modes = {mode for mode, _ in items}
    assert "updates" in modes
    assert "messages" in modes
