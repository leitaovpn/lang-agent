"""AgentLoop（ReAct 循环）行为测试，全部用 FakeChatModel，不依赖真实 API。"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from lang_agent.core.react_agent import AgentLoop, AgentLoopConfig, build_checkpointer
from lang_agent.core.retry import RetryableError
from tests.conftest import FakeChatModel, FlakyChatModel

TOOL_CALL = {
    "name": "calculator",
    "args": {"expression": "(3+5)*7"},
    "id": "call_1",
    "type": "tool_call",
}


def make_loop(script, *, checkpointer=None, config=None):
    llm = FakeChatModel(responses=list(script))
    loop = AgentLoop(llm=llm, config=config, checkpointer=checkpointer)
    return loop, llm


async def test_invoke_plain_text():
    loop, _ = make_loop([AIMessage(content="答案是 42")])
    result = await loop.invoke("1+1 等于几", thread_id="t1")
    assert result.final_text == "答案是 42"
    assert result.tool_calls == []
    assert result.messages[-1].content == "答案是 42"


async def test_invoke_with_tool_round_executes_real_tool():
    loop, _ = make_loop([AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="结果是 56")])
    result = await loop.invoke("计算 (3+5)*7", thread_id="t1")
    assert result.final_text == "结果是 56"
    assert [tc["name"] for tc in result.tool_calls] == ["calculator"]
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert tool_messages and tool_messages[0].content == "56"


async def test_memory_second_round_sees_history():
    loop, llm = make_loop(
        [AIMessage(content="第一次回答"), AIMessage(content="第二次回答")],
        checkpointer=InMemorySaver(),
    )
    await loop.invoke("第一问", thread_id="t1")
    await loop.invoke("第二问", thread_id="t1")
    seen = llm.seen_messages[1]
    assert len(seen) >= 3
    assert isinstance(seen[0], HumanMessage) and seen[0].content == "第一问"
    assert isinstance(seen[-1], HumanMessage) and seen[-1].content == "第二问"


async def test_different_thread_ids_isolated():
    loop, llm = make_loop(
        [AIMessage(content="回答1"), AIMessage(content="回答2")],
        checkpointer=InMemorySaver(),
    )
    await loop.invoke("问", thread_id="t1")
    await loop.invoke("问", thread_id="t2")
    for seen in llm.seen_messages:
        assert len(seen) == 1
        assert isinstance(seen[0], HumanMessage) and seen[0].content == "问"


async def test_system_message_prepended():
    loop, llm = make_loop([AIMessage(content="好")])
    await loop.invoke("问", thread_id="t1", system="你是计算助手")
    assert llm.seen_messages[0][0] == SystemMessage(content="你是计算助手")


async def test_stream_events_order_and_content():
    loop, _ = make_loop([AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="结果是 56")])
    events = [event async for event in loop.stream("计算 (3+5)*7", thread_id="t1")]
    types = [e.type for e in events]
    assert "llm_token" in types
    assert types.index("tool_call") < types.index("tool_result") < types.index("done")
    done = events[-1]
    assert done.type == "done"
    assert done.data["final_text"] == "结果是 56"
    tool_result = next(e for e in events if e.type == "tool_result")
    assert tool_result.data["content"] == "56"


async def test_stream_emits_token_level_llm_events():
    # 最终回答必须按 token 逐条产出（而不是节点结束时整段文本一条）
    loop, _ = make_loop([AIMessage(content="结果是 56")])
    events = [event async for event in loop.stream("计算", thread_id="t1")]
    tokens = [e.data["text"] for e in events if e.type == "llm_token"]
    assert "".join(tokens) == "结果是 56"
    assert len(tokens) > 1


async def test_stream_error_yields_error_event():
    class BoomChatModel(FakeChatModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("模型挂了")
            yield  # pragma: no cover

    boom = BoomChatModel(responses=[AIMessage(content="x")])
    bad_loop = AgentLoop(llm=boom)
    events = [event async for event in bad_loop.stream("问", thread_id="t1")]
    assert events[-1].type == "error"
    assert "模型挂了" in events[-1].data["message"]


async def test_sqlite_checkpointer_persists_across_instances(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_path = str(tmp_path / "ck.sqlite")
    config = AgentLoopConfig(checkpointer_kind="sqlite", db_path=db_path)

    # async with 保证连接关闭，否则 aiosqlite 连接会让测试会话挂起
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver1:
        loop1, _ = make_loop([AIMessage(content="第一次回答")], config=config, checkpointer=saver1)
        await loop1.invoke("第一问", thread_id="t1")

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver2:
        loop2, llm2 = make_loop([AIMessage(content="第二次回答")], config=config, checkpointer=saver2)
        await loop2.invoke("第二问", thread_id="t1")
        seen = llm2.seen_messages[0]
        assert isinstance(seen[0], HumanMessage) and seen[0].content == "第一问"
        assert isinstance(seen[-1], HumanMessage) and seen[-1].content == "第二问"


async def test_tool_calls_only_summarize_current_round():
    # 第一轮触发工具调用，第二轮纯文本：第二轮结果不应包含第一轮的 tool_calls
    loop, _ = make_loop(
        [
            AIMessage(content="", tool_calls=[TOOL_CALL]),   # 第 1 轮
            AIMessage(content="结果是 56"),                    # 第 1 轮
            AIMessage(content="好的，没问题"),                  # 第 2 轮
        ],
        checkpointer=InMemorySaver(),
    )
    first = await loop.invoke("计算 (3+5)*7", thread_id="t1")
    assert [tc["name"] for tc in first.tool_calls] == ["calculator"]

    second = await loop.invoke("谢谢", thread_id="t1")
    assert second.tool_calls == []


async def test_repair_persists_to_checkpoint():
    # 第一轮模型返回重复 id 的 tool_call，产生坏尾部；第二轮调用时入口修复
    # 把坏段写回 checkpoint——之后拉取到的历史 id 唯一、ToolMessage 一一对应
    calls = [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup", "type": "tool_call"},
    ]
    loop, _ = make_loop(
        [
            AIMessage(content="", tool_calls=calls),
            AIMessage(content="分别是 2 和 4"),
            AIMessage(content="好的"),
        ],
        checkpointer=InMemorySaver(),
    )
    await loop.invoke("算两个", thread_id="t1")
    await loop.invoke("谢谢", thread_id="t1")

    state = await loop._graph.aget_state({"configurable": {"thread_id": "t1"}})
    messages = state.values["messages"]
    ai = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls][0]
    ids = [c["id"] for c in ai.tool_calls]
    assert len(ids) == len(set(ids)) == 2
    tool_ids = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert sorted(tool_ids) == sorted(ids)


async def test_result_messages_only_current_round():
    loop, _ = make_loop(
        [AIMessage(content="第一答"), AIMessage(content="第二答")],
        checkpointer=InMemorySaver(),
    )
    first = await loop.invoke("第一问", thread_id="t1")
    assert [m.content for m in first.messages] == ["第一问", "第一答"]

    second = await loop.invoke("第二问", thread_id="t1")
    assert [m.content for m in second.messages] == ["第二问", "第二答"]


async def test_stream_done_only_summarizes_current_round_tool_calls():
    loop, _ = make_loop(
        [
            AIMessage(content="", tool_calls=[TOOL_CALL]),
            AIMessage(content="结果是 56"),
            AIMessage(content="好的"),
        ],
        checkpointer=InMemorySaver(),
    )
    await loop.invoke("计算 (3+5)*7", thread_id="t1")
    events = [event async for event in loop.stream("谢谢", thread_id="t1")]
    done = events[-1]
    assert done.type == "done"
    assert done.data["tool_calls"] == []


async def test_loop_retries_on_invalid_tool_calls():
    # 第一轮返回解析失败的 tool_call → 循环应附错误反馈并重试，最终给出回答
    invalid = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "calculator", "args": "{bad", "id": None, "error": "JSON 解析失败"}
        ],
    )
    loop, llm = make_loop([invalid, AIMessage(content="抱歉，我重新算：答案是 42")])
    result = await loop.invoke("算一下 1+1", thread_id="t1")
    assert result.final_text == "抱歉，我重新算：答案是 42"
    # 第二轮 LLM 看到的历史里包含错误反馈 ToolMessage
    seen = llm.seen_messages[1]
    assert any(isinstance(m, ToolMessage) and "格式错误" in m.content for m in seen)


async def test_loop_completes_with_duplicate_tool_call_ids():
    # 模型返回重复 id 的 tool_call：工具照常执行、循环照常完成；
    # 历史修复由下次调用的入口 checkpoint 修复兜底
    calls = [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup", "type": "tool_call"},
    ]
    loop, llm = make_loop([AIMessage(content="", tool_calls=calls), AIMessage(content="分别是 2 和 4")])
    result = await loop.invoke("算两个", thread_id="t1")
    assert result.final_text == "分别是 2 和 4"
    tool_contents = sorted(m.content for m in result.messages if isinstance(m, ToolMessage))
    assert tool_contents == ["2", "4"]
    # 不做 LLM 拷贝修复：第二轮模型看到的历史保持原样（重复 id 原样传入）
    seen = llm.seen_messages[1]
    ai = [m for m in seen if isinstance(m, AIMessage) and m.tool_calls][0]
    assert [c["id"] for c in ai.tool_calls] == ["dup", "dup"]


def make_flaky_loop(script, *, fail_times=1, fail_at=None, error_factory=None, error=None, config=None):
    """构造 FlakyChatModel 驱动的 loop；测试用 retry_base_delay=0 避免真实等待。"""
    flaky = FlakyChatModel(
        responses=list(script),
        fail_times=fail_times,
        fail_at=fail_at,
        error=error,
        error_factory=error_factory,
    )
    cfg = config or AgentLoopConfig(retry_base_delay=0)
    return AgentLoop(llm=flaky, config=cfg, checkpointer=InMemorySaver()), flaky


async def test_invoke_retries_transient_error_without_duplicate_messages():
    loop, flaky = make_flaky_loop(
        [AIMessage(content="答案是 42")],
        fail_times=1,
        error_factory=lambda: RetryableError("瞬时限流"),
    )
    result = await loop.invoke("1+1 等于几", thread_id="t1")
    assert result.final_text == "答案是 42"
    # 重试从 checkpoint 续跑：失败的超步重执行，输入消息不重复追加
    assert flaky.calls == 2
    for seen in flaky.seen_messages:
        humans = [m for m in seen if isinstance(m, HumanMessage)]
        assert len(humans) == 1 and humans[0].content == "1+1 等于几"


async def test_invoke_raises_after_max_attempts():
    loop, flaky = make_flaky_loop(
        [AIMessage(content="不会用到")],
        fail_times=10,
        error_factory=lambda: RetryableError("持续限流"),
        config=AgentLoopConfig(retry_base_delay=0, retry_max_attempts=3),
    )
    with pytest.raises(RetryableError):
        await loop.invoke("问", thread_id="t1")
    assert flaky.calls == 3


async def test_invoke_non_retryable_error_raises_immediately():
    loop, flaky = make_flaky_loop(
        [AIMessage(content="不会用到")],
        error_factory=lambda: RuntimeError("确定性错误"),
    )
    with pytest.raises(RuntimeError):
        await loop.invoke("问", thread_id="t1")
    assert flaky.calls == 1


async def test_stream_retries_transient_error_and_finishes_done():
    loop, flaky = make_flaky_loop(
        [AIMessage(content="结果是 56")],
        fail_times=1,
        error_factory=lambda: RetryableError("瞬时超时"),
    )
    events = [event async for event in loop.stream("计算", thread_id="t1")]
    assert events[-1].type == "done"
    assert events[-1].data["final_text"] == "结果是 56"
    assert flaky.calls == 2


async def test_retry_resume_repairs_mid_run_broken_messages():
    # 第 1 次调用返回重复 id 的 tool_call（中途产生坏段），第 2 次调用瞬时失败，
    # 续跑前的修复必须让续跑调用看到 tool_call id 唯一、ToolMessage 一一对应的历史。
    # 注意：失败的调用在 _generate 之前就抛异常，不记入 seen_messages。
    dup_calls = [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup", "type": "tool_call"},
    ]
    loop, flaky = make_flaky_loop(
        [AIMessage(content="", tool_calls=dup_calls), AIMessage(content="答案是 X")],
        fail_times=0,
        fail_at=2,
        error_factory=lambda: RetryableError("第二轮流式中断"),
    )
    result = await loop.invoke("算两个", thread_id="t1")
    assert result.final_text == "答案是 X"
    assert flaky.calls == 3
    # seen[0]：第 1 次调用（只看到用户消息）；seen[1]：续跑调用（看到修复后的历史）
    resumed = flaky.seen_messages[1]
    ai_resumed = [m for m in resumed if isinstance(m, AIMessage) and m.tool_calls][0]
    ids = [c["id"] for c in ai_resumed.tool_calls]
    assert len(ids) == len(set(ids)) == 2
    tool_ids = [m.tool_call_id for m in resumed if isinstance(m, ToolMessage)]
    assert sorted(tool_ids) == sorted(ids)


async def test_stream_retry_resume_repairs_mid_run_broken_messages():
    dup_calls = [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup", "type": "tool_call"},
    ]
    loop, flaky = make_flaky_loop(
        [AIMessage(content="", tool_calls=dup_calls), AIMessage(content="答案是 X")],
        fail_times=0,
        fail_at=2,
        error_factory=lambda: RetryableError("第二轮流式中断"),
    )
    events = [event async for event in loop.stream("算两个", thread_id="t1")]
    assert events[-1].type == "done"
    assert events[-1].data["final_text"] == "答案是 X"
    resumed = flaky.seen_messages[1]
    ai_resumed = [m for m in resumed if isinstance(m, AIMessage) and m.tool_calls][0]
    ids = [c["id"] for c in ai_resumed.tool_calls]
    assert len(ids) == len(set(ids)) == 2
    tool_ids = [m.tool_call_id for m in resumed if isinstance(m, ToolMessage)]
    assert sorted(tool_ids) == sorted(ids)


async def test_stream_ends_with_error_event_after_exhaustion():
    loop, _ = make_flaky_loop(
        [AIMessage(content="不会用到")],
        fail_times=10,
        error_factory=lambda: RetryableError("持续超时"),
        config=AgentLoopConfig(retry_base_delay=0, retry_max_attempts=2),
    )
    events = [event async for event in loop.stream("问", thread_id="t1")]
    assert events[-1].type == "error"
    assert "持续超时" in events[-1].data["message"]


# ---- ToolNode 异常语义锁定：工具异常/未知工具 → 错误 ToolMessage 投喂 LLM，不抛出 ----


async def test_tool_execution_error_feeds_error_to_llm():
    # 工具执行异常（除零）：ToolNode 捕获后以错误 ToolMessage 投喂 LLM，循环继续
    calls = [
        {"name": "calculator", "args": {"expression": "1/0"}, "id": "c1", "type": "tool_call"}
    ]
    loop, _ = make_loop([AIMessage(content="", tool_calls=calls), AIMessage(content="除零失败，换个算法")])
    result = await loop.invoke("1 除以 0", thread_id="t1")
    assert result.final_text == "除零失败，换个算法"
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert "无法计算" in tool_msgs[0].content or "division by zero" in tool_msgs[0].content
    assert tool_msgs[0].status == "error"


async def test_unknown_tool_feeds_error_to_llm():
    # LLM 调用 ToolNode 范围外的工具：返回含报错信息的 ToolMessage，循环继续
    calls = [
        {"name": "no_such_tool", "args": {"x": 1}, "id": "c1", "type": "tool_call"}
    ]
    loop, _ = make_loop([AIMessage(content="", tool_calls=calls), AIMessage(content="没有这个工具，我直接回答")])
    result = await loop.invoke("用不存在的工具", thread_id="t1")
    assert result.final_text == "没有这个工具，我直接回答"
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert "not a valid tool" in tool_msgs[0].content


async def test_stream_tool_execution_error_feeds_error_to_llm():
    calls = [
        {"name": "calculator", "args": {"expression": "1/0"}, "id": "c1", "type": "tool_call"}
    ]
    loop, _ = make_loop([AIMessage(content="", tool_calls=calls), AIMessage(content="除零失败，换个算法")])
    events = [event async for event in loop.stream("1 除以 0", thread_id="t1")]
    types = [e.type for e in events]
    assert types.index("tool_call") < types.index("tool_result") < types.index("done")
    tool_result = next(e for e in events if e.type == "tool_result")
    assert "无法计算" in tool_result.data["content"] or "division by zero" in tool_result.data["content"]
    assert events[-1].data["final_text"] == "除零失败，换个算法"


async def test_stream_unknown_tool_feeds_error_to_llm():
    calls = [
        {"name": "no_such_tool", "args": {"x": 1}, "id": "c1", "type": "tool_call"}
    ]
    loop, _ = make_loop([AIMessage(content="", tool_calls=calls), AIMessage(content="没有这个工具，我直接回答")])
    events = [event async for event in loop.stream("用不存在的工具", thread_id="t1")]
    types = [e.type for e in events]
    assert types.index("tool_call") < types.index("tool_result") < types.index("done")
    tool_result = next(e for e in events if e.type == "tool_result")
    assert "not a valid tool" in tool_result.data["content"]
    assert events[-1].data["final_text"] == "没有这个工具，我直接回答"


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
    assert loop._graph.checkpointer is not None
    assert loop._graph.checkpointer.__class__.__name__ in ("InMemorySaver", "MemorySaver")
