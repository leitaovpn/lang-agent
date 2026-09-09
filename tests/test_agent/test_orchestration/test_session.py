"""ChatSession（会话入口：repair/compress/retry/塑形/事件分类）行为测试。

全部用 FakeChatModel，不依赖真实 API；graph 直驱行为见 tests/test_core/test_loop/test_react_agent.py。
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from lang_agent.agent.orchestration import ChatSession
from lang_agent.core.loop import AgentContext, AgentLoop, AgentLoopConfig
from lang_agent.core.loop.retry import RetryableError
from lang_agent.core.tool import instantiate_tools
from tests.conftest import FakeChatModel, FlakyChatModel

TOOL_CALL = {
    "name": "calculator",
    "args": {"expression": "(3+5)*7"},
    "id": "call_1",
    "type": "tool_call",
}


def make_ctx(llm, tools=None, summarizer_llm=None) -> AgentContext:
    return AgentContext(
        llm=llm,
        tools=tools if tools is not None else instantiate_tools(),
        summarizer_llm=summarizer_llm,
    )


def make_session(script, *, checkpointer=None, config=None):
    llm = FakeChatModel(responses=list(script))
    loop = AgentLoop(config=config, checkpointer=checkpointer)
    return ChatSession(loop=loop, config=config), llm, make_ctx(llm)


def make_flaky_session(
    script,
    *,
    fail_times=1,
    fail_at=None,
    error_factory=None,
    error=None,
    config=None,
):
    """构造 FlakyChatModel 驱动的 session；测试用 retry_base_delay=0 避免真实等待。"""
    flaky = FlakyChatModel(
        responses=list(script),
        fail_times=fail_times,
        fail_at=fail_at,
        error=error,
        error_factory=error_factory,
    )
    cfg = config or AgentLoopConfig(retry_base_delay=0)
    loop = AgentLoop(config=cfg, checkpointer=InMemorySaver())
    return ChatSession(loop=loop, config=cfg), flaky, make_ctx(flaky)


async def test_invoke_plain_text():
    session, _, ctx = make_session([AIMessage(content="答案是 42")])
    result = await session.invoke("1+1 等于几", thread_id="t1", context=ctx)
    assert result.final_text == "答案是 42"
    assert result.tool_calls == []
    assert result.messages[-1].content == "答案是 42"


async def test_invoke_with_tool_round_executes_real_tool():
    session, _, ctx = make_session(
        [AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="结果是 56")]
    )
    result = await session.invoke("计算 (3+5)*7", thread_id="t1", context=ctx)
    assert result.final_text == "结果是 56"
    assert [tc["name"] for tc in result.tool_calls] == ["calculator"]
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert tool_messages and tool_messages[0].content == "56"


async def test_tool_calls_only_summarize_current_round():
    # 第一轮触发工具调用，第二轮纯文本：第二轮结果不应包含第一轮的 tool_calls
    session, _, ctx = make_session(
        [
            AIMessage(content="", tool_calls=[TOOL_CALL]),   # 第 1 轮
            AIMessage(content="结果是 56"),                    # 第 1 轮
            AIMessage(content="好的，没问题"),                  # 第 2 轮
        ],
        checkpointer=InMemorySaver(),
    )
    first = await session.invoke("计算 (3+5)*7", thread_id="t1", context=ctx)
    assert [tc["name"] for tc in first.tool_calls] == ["calculator"]

    second = await session.invoke("谢谢", thread_id="t1", context=ctx)
    assert second.tool_calls == []


async def test_result_messages_only_current_round():
    session, _, ctx = make_session(
        [AIMessage(content="第一答"), AIMessage(content="第二答")],
        checkpointer=InMemorySaver(),
    )
    first = await session.invoke("第一问", thread_id="t1", context=ctx)
    assert [m.content for m in first.messages] == ["第一问", "第一答"]

    second = await session.invoke("第二问", thread_id="t1", context=ctx)
    assert [m.content for m in second.messages] == ["第二问", "第二答"]


async def test_stream_events_order_and_content():
    session, _, ctx = make_session(
        [AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="结果是 56")]
    )
    events = [event async for event in session.stream("计算 (3+5)*7", thread_id="t1", context=ctx)]
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
    session, _, ctx = make_session([AIMessage(content="结果是 56")])
    events = [event async for event in session.stream("计算", thread_id="t1", context=ctx)]
    tokens = [e.data["text"] for e in events if e.type == "llm_token"]
    assert "".join(tokens) == "结果是 56"
    assert len(tokens) > 1


async def test_invalid_tool_call_round_history_consistent():
    """第一轮 invalid 工具调用（解析失败）后，第二轮发往 LLM 的历史必须满足
    「每个 tool_calls.id 都有匹配的 ToolMessage」——agent_node 与 repair 的
    invalid id 必须统一为 invalid_<aimessage下标>_<条内序号>，否则 deepseek 报 400。
    """
    invalid_call = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "calculator", "args": "(3+5)*", "id": "call_9", "type": "invalid_tool_call", "error": None}
        ],
    )
    # invalid 轮循环：第 1 次 invalid → 反馈 ToolMessage → 第 2 次修正重试成功
    session, llm, ctx = make_session(
        [invalid_call, AIMessage(content="重试成功"), AIMessage(content="继续回答")],
        checkpointer=InMemorySaver(),
    )
    await session.invoke("计算 (3+5)*", thread_id="t1", context=ctx)
    await session.invoke("继续", thread_id="t1", context=ctx)
    # 第二轮 LLM 历史：invalid 段（AIMessage 的 invalid_tool_calls）第 1 次合成的
    # 反馈 ToolMessage 必须在 repair 后仍用确定性 invalid_ 前缀 id（与发送给
    # API 的 tool_calls id 一致），不得用原始 id call_9 —— 否则 deepseek 报 400。
    second_round_history = llm.seen_messages[-1]
    invalid_feedbacks = [
        m for m in second_round_history if isinstance(m, ToolMessage) and m.tool_call_id.startswith("invalid_")
    ]
    assert invalid_feedbacks, "invalid 反馈 ToolMessage 缺失"
    # 每个 invalid_tool_calls 的 id 也必须与反馈 ToolMessage 的 tool_call_id 一致
    # （_convert_message_to_dict 将 invalid_tool_calls 序列化为 tool_calls 发送）
    for msg in second_round_history:
        if isinstance(msg, AIMessage):
            for inv in msg.invalid_tool_calls or []:
                assert any(
                    tm.tool_call_id == inv["id"]
                    for tm in invalid_feedbacks
                ), f"invalid_tool_calls id {inv['id']} 无匹配 ToolMessage（原始 id 需重写为 invalid_ 前缀）"
    # 第二轮 LLM 收到的历史：每条 tool_calls 都必须被 ToolMessage 覆盖
    history = llm.seen_messages[-1]
    for msg in history:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                assert any(
                    isinstance(prev, ToolMessage) and prev.tool_call_id == call["id"]
                    for prev in history
                ), f"tool_calls id {call['id']} 无匹配 ToolMessage"


async def test_stream_emits_thinking_before_answer():
    # deepseek 系模型的 reasoning_content 增量：thinking_token 在 llm_token 之前逐 token 产出
    thinking = AIMessage(
        content="结果是 56",
        additional_kwargs={"reasoning_content": "先想想"},
    )
    session, _, ctx = make_session([thinking])
    events = [event async for event in session.stream("计算", thread_id="t1", context=ctx)]
    thinking_tokens = [e.data["text"] for e in events if e.type == "thinking_token"]
    answer_tokens = [e.data["text"] for e in events if e.type == "llm_token"]
    assert "".join(thinking_tokens) == "先想想"
    assert "".join(answer_tokens) == "结果是 56"
    assert events.index(next(e for e in events if e.type == "thinking_token")) < events.index(
        next(e for e in events if e.type == "llm_token")
    )


async def test_stream_thinking_before_tool_round():
    # 含工具轮时每轮思考都产出 thinking_token（repair 不变式不受影响）
    thinking_call = AIMessage(
        content="",
        tool_calls=[TOOL_CALL],
        additional_kwargs={"reasoning_content": "先想"},
    )
    final = AIMessage(
        content="结果是 56",
        additional_kwargs={"reasoning_content": "再想"},
    )
    session, _, ctx = make_session([thinking_call, final])
    events = [event async for event in session.stream("计算", thread_id="t1", context=ctx)]
    assert sum(1 for e in events if e.type == "thinking_token") > 0
    assert "".join(e.data["text"] for e in events if e.type == "thinking_token") == "先想再想"


async def test_stream_done_only_summarizes_current_round_tool_calls():
    session, _, ctx = make_session(
        [
            AIMessage(content="", tool_calls=[TOOL_CALL]),
            AIMessage(content="结果是 56"),
            AIMessage(content="好的"),
        ],
        checkpointer=InMemorySaver(),
    )
    await session.invoke("计算 (3+5)*7", thread_id="t1", context=ctx)
    events = [event async for event in session.stream("谢谢", thread_id="t1", context=ctx)]
    done = events[-1]
    assert done.type == "done"
    assert done.data["tool_calls"] == []


async def test_stream_error_yields_error_event():
    class BoomChatModel(FakeChatModel):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("模型挂了")
            yield  # pragma: no cover

    boom = BoomChatModel(responses=[AIMessage(content="x")])
    session = ChatSession(loop=AgentLoop(checkpointer=InMemorySaver()))
    events = [
        event
        async for event in session.stream("问", thread_id="t1", context=make_ctx(boom))
    ]
    assert events[-1].type == "error"
    assert "模型挂了" in events[-1].data["message"]


async def test_repair_persists_to_checkpoint():
    # 第一轮模型返回重复 id 的 tool_call，产生坏尾部；第二轮调用时入口修复
    # 把坏段写回 checkpoint——之后拉取到的历史 id 唯一、ToolMessage 一一对应
    calls = [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup", "type": "tool_call"},
    ]
    session, _, ctx = make_session(
        [
            AIMessage(content="", tool_calls=calls),
            AIMessage(content="分别是 2 和 4"),
            AIMessage(content="好的"),
        ],
        checkpointer=InMemorySaver(),
    )
    await session.invoke("算两个", thread_id="t1", context=ctx)
    await session.invoke("谢谢", thread_id="t1", context=ctx)

    state = await session.loop.graph.aget_state({"configurable": {"thread_id": "t1"}})
    messages = state.values["messages"]
    ai = next(m for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    ids = [c["id"] for c in ai.tool_calls]
    assert len(ids) == len(set(ids)) == 2
    tool_ids = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert sorted(tool_ids) == sorted(ids)


async def test_invoke_retries_transient_error_without_duplicate_messages():
    session, flaky, ctx = make_flaky_session(
        [AIMessage(content="答案是 42")],
        fail_times=1,
        error_factory=lambda: RetryableError("瞬时限流"),
    )
    result = await session.invoke("1+1 等于几", thread_id="t1", context=ctx)
    assert result.final_text == "答案是 42"
    # 重试从 checkpoint 续跑：失败的超步重执行，输入消息不重复追加
    assert flaky.calls == 2
    for seen in flaky.seen_messages:
        humans = [m for m in seen if isinstance(m, HumanMessage)]
        assert len(humans) == 1 and humans[0].content == "1+1 等于几"


async def test_invoke_raises_after_max_attempts():
    session, flaky, ctx = make_flaky_session(
        [AIMessage(content="不会用到")],
        fail_times=10,
        error_factory=lambda: RetryableError("持续限流"),
        config=AgentLoopConfig(retry_base_delay=0, retry_max_attempts=3),
    )
    with pytest.raises(RetryableError):
        await session.invoke("问", thread_id="t1", context=ctx)
    assert flaky.calls == 3


async def test_invoke_non_retryable_error_raises_immediately():
    session, flaky, ctx = make_flaky_session(
        [AIMessage(content="不会用到")],
        error_factory=lambda: RuntimeError("确定性错误"),
    )
    with pytest.raises(RuntimeError):
        await session.invoke("问", thread_id="t1", context=ctx)
    assert flaky.calls == 1


async def test_stream_retries_transient_error_and_finishes_done():
    session, flaky, ctx = make_flaky_session(
        [AIMessage(content="结果是 56")],
        fail_times=1,
        error_factory=lambda: RetryableError("瞬时超时"),
    )
    events = [event async for event in session.stream("计算", thread_id="t1", context=ctx)]
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
    session, flaky, ctx = make_flaky_session(
        [AIMessage(content="", tool_calls=dup_calls), AIMessage(content="答案是 X")],
        fail_times=0,
        fail_at=2,
        error_factory=lambda: RetryableError("第二轮流式中断"),
    )
    result = await session.invoke("算两个", thread_id="t1", context=ctx)
    assert result.final_text == "答案是 X"
    assert flaky.calls == 3
    # seen[0]：第 1 次调用（只看到用户消息）；seen[1]：续跑调用（看到修复后的历史）
    resumed = flaky.seen_messages[1]
    ai_resumed = next(m for m in resumed if isinstance(m, AIMessage) and m.tool_calls)
    ids = [c["id"] for c in ai_resumed.tool_calls]
    assert len(ids) == len(set(ids)) == 2
    tool_ids = [m.tool_call_id for m in resumed if isinstance(m, ToolMessage)]
    assert sorted(tool_ids) == sorted(ids)


async def test_stream_retry_resume_repairs_mid_run_broken_messages():
    dup_calls = [
        {"name": "calculator", "args": {"expression": "1+1"}, "id": "dup", "type": "tool_call"},
        {"name": "calculator", "args": {"expression": "2+2"}, "id": "dup", "type": "tool_call"},
    ]
    session, flaky, ctx = make_flaky_session(
        [AIMessage(content="", tool_calls=dup_calls), AIMessage(content="答案是 X")],
        fail_times=0,
        fail_at=2,
        error_factory=lambda: RetryableError("第二轮流式中断"),
    )
    events = [event async for event in session.stream("算两个", thread_id="t1", context=ctx)]
    assert events[-1].type == "done"
    assert events[-1].data["final_text"] == "答案是 X"
    resumed = flaky.seen_messages[1]
    ai_resumed = next(m for m in resumed if isinstance(m, AIMessage) and m.tool_calls)
    ids = [c["id"] for c in ai_resumed.tool_calls]
    assert len(ids) == len(set(ids)) == 2
    tool_ids = [m.tool_call_id for m in resumed if isinstance(m, ToolMessage)]
    assert sorted(tool_ids) == sorted(ids)


async def test_stream_ends_with_error_event_after_exhaustion():
    session, _, ctx = make_flaky_session(
        [AIMessage(content="不会用到")],
        fail_times=10,
        error_factory=lambda: RetryableError("持续超时"),
        config=AgentLoopConfig(retry_base_delay=0, retry_max_attempts=2),
    )
    events = [event async for event in session.stream("问", thread_id="t1", context=ctx)]
    assert events[-1].type == "error"
    assert "持续超时" in events[-1].data["message"]


# ---- LLM/tools 经 context 注入：每次 run 可替换（session 每次调用显式传） ----


async def test_invoke_uses_custom_context_llm_and_tools():
    session, _, ctx = make_session([AIMessage(content="默认上下文回答")])

    custom_llm = FakeChatModel(responses=[AIMessage(content="自定义上下文回答")])
    custom_ctx = make_ctx(custom_llm)
    result = await session.invoke("你好", thread_id="t1", context=custom_ctx)
    assert result.final_text == "自定义上下文回答"
    assert custom_llm.seen_messages
    assert not ctx.llm.seen_messages


async def test_custom_context_restricts_tools():
    # context.tools 只给 calculator：LLM 调 string_len 应得到 not a valid tool 错误反馈
    calls = [
        {"name": "string_len", "args": {"text": "abc"}, "id": "c1", "type": "tool_call"}
    ]
    llm = FakeChatModel(responses=[AIMessage(content="", tool_calls=calls), AIMessage(content="工具不可用，我直接回答")])
    session = ChatSession(loop=AgentLoop(checkpointer=InMemorySaver()))

    calculator_only = [t for t in instantiate_tools() if t.name == "calculator"]
    result = await session.invoke(
        "算 abc 长度", thread_id="t1", context=AgentContext(llm=llm, tools=calculator_only)
    )
    assert result.final_text == "工具不可用，我直接回答"
    tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "not a valid tool" in tool_msgs[0].content


async def test_stream_uses_custom_context():
    session, _, ctx = make_session([AIMessage(content="默认上下文回答")])

    custom_llm = FakeChatModel(responses=[AIMessage(content="流式自定义回答")])
    events = [
        event
        async for event in session.stream(
            "你好", thread_id="t1", context=make_ctx(custom_llm)
        )
    ]
    assert events[-1].type == "done"
    assert events[-1].data["final_text"] == "流式自定义回答"
    assert not ctx.llm.seen_messages


# ---- context 压缩：摘要 + 保留窗口 + 工具输出截断 ----


async def test_compress_writes_back_summary_and_removes_old_rounds():
    summarizer = FakeChatModel(responses=[AIMessage(content="早期摘要内容")])
    llm = FakeChatModel(responses=[AIMessage(content="答1"), AIMessage(content="答2"), AIMessage(content="答3"), AIMessage(content="答4")])
    session, _, _ = make_session(
        [AIMessage(content="答1"), AIMessage(content="答2"), AIMessage(content="答3"), AIMessage(content="答4")],
        config=AgentLoopConfig(compress_token_threshold=1, compress_keep_last=4),
        checkpointer=InMemorySaver(),
    )
    ctx = AgentContext(llm=llm, tools=instantiate_tools(), summarizer_llm=summarizer)
    for q in ["问1", "问2", "问3", "问4"]:
        await session.invoke(q, thread_id="t1", context=ctx)
        # 第 4 轮入口才发生压缩（入口时历史 6 条 > keep_last 4）

    state = await session.loop.graph.aget_state({"configurable": {"thread_id": "t1"}})
    values = state.values
    assert values.get("summary") == "早期摘要内容"
    contents = [m.content for m in values["messages"] if isinstance(m, (HumanMessage, AIMessage))]
    assert contents == ["问2", "答2", "问3", "答3", "问4", "答4"]
    # 第 4 轮 LLM 收到摘要前缀
    seen = llm.seen_messages[3]
    assert any(isinstance(m, SystemMessage) and "早期对话摘要" in m.content and "早期摘要内容" in m.content for m in seen)


async def test_compress_falls_back_to_dialogue_llm_as_summarizer():
    # summarizer_llm 缺省时复用对话 llm：脚本需含摘要消费的条目
    llm = FakeChatModel(
        responses=[
            AIMessage(content="答1"),
            AIMessage(content="答2"),
            AIMessage(content="答3"),
            AIMessage(content="对话llm做的摘要"),
            AIMessage(content="答4"),
        ]
    )
    session, _, _ = make_session(
        [
            AIMessage(content="答1"),
            AIMessage(content="答2"),
            AIMessage(content="答3"),
            AIMessage(content="对话llm做的摘要"),
            AIMessage(content="答4"),
        ],
        config=AgentLoopConfig(compress_token_threshold=1, compress_keep_last=4),
        checkpointer=InMemorySaver(),
    )
    ctx = AgentContext(llm=llm, tools=instantiate_tools())
    for q in ["问1", "问2", "问3", "问4"]:
        await session.invoke(q, thread_id="t1", context=ctx)
    state = await session.loop.graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert state.values.get("summary") == "对话llm做的摘要"


async def test_compress_keeps_repair_invariant():
    summarizer = FakeChatModel(responses=[AIMessage(content="摘要")])
    llm = FakeChatModel(responses=[AIMessage(content="答1"), AIMessage(content="答2"), AIMessage(content="答3"), AIMessage(content="答4")])
    session, _, _ = make_session(
        [AIMessage(content="答1"), AIMessage(content="答2"), AIMessage(content="答3"), AIMessage(content="答4")],
        config=AgentLoopConfig(compress_token_threshold=1, compress_keep_last=4),
        checkpointer=InMemorySaver(),
    )
    ctx = AgentContext(llm=llm, tools=instantiate_tools(), summarizer_llm=summarizer)
    for q in ["问1", "问2", "问3", "问4"]:
        await session.invoke(q, thread_id="t1", context=ctx)

    from lang_agent.core.loop.repair import repair_state_for_checkpoint

    state = await session.loop.graph.aget_state({"configurable": {"thread_id": "t1"}})
    messages = list(state.values["messages"])
    assert repair_state_for_checkpoint(messages) == messages  # 压缩后历史仍满足不变式


async def test_compress_skipped_below_threshold():
    llm = FakeChatModel(responses=[AIMessage(content="答1"), AIMessage(content="答2")])
    session, _, _ = make_session(
        [AIMessage(content="答1"), AIMessage(content="答2")],
        config=AgentLoopConfig(compress_token_threshold=10**9),
        checkpointer=InMemorySaver(),
    )
    ctx = make_ctx(llm)
    await session.invoke("问1", thread_id="t1", context=ctx)
    await session.invoke("问2", thread_id="t1", context=ctx)
    state = await session.loop.graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert (state.values.get("summary") or "") == ""
    contents = [m.content for m in state.values["messages"] if isinstance(m, (HumanMessage, AIMessage))]
    assert contents == ["问1", "答1", "问2", "答2"]


async def test_compress_uses_dedicated_summarizer():
    summarizer = FakeChatModel(responses=[AIMessage(content="独立摘要")])
    llm = FakeChatModel(responses=[AIMessage(content="答1"), AIMessage(content="答2"), AIMessage(content="答3"), AIMessage(content="答4")])
    session, _, _ = make_session(
        [AIMessage(content="答1"), AIMessage(content="答2"), AIMessage(content="答3"), AIMessage(content="答4")],
        config=AgentLoopConfig(compress_token_threshold=1, compress_keep_last=4),
        checkpointer=InMemorySaver(),
    )
    ctx = AgentContext(llm=llm, tools=instantiate_tools(), summarizer_llm=summarizer)
    for q in ["问1", "问2", "问3", "问4"]:
        await session.invoke(q, thread_id="t1", context=ctx)
    state = await session.loop.graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert state.values.get("summary") == "独立摘要"
    assert summarizer.seen_messages  # 独立摘要模型承担了摘要
    assert len(llm.seen_messages) == 4  # 对话 llm 未被摘要调用消耗


# ---- ToolNode 异常语义（流式通道） ----


async def test_stream_tool_execution_error_feeds_error_to_llm():
    calls = [
        {"name": "calculator", "args": {"expression": "1/0"}, "id": "c1", "type": "tool_call"}
    ]
    session, _, ctx = make_session(
        [AIMessage(content="", tool_calls=calls), AIMessage(content="除零失败，换个算法")]
    )
    events = [event async for event in session.stream("1 除以 0", thread_id="t1", context=ctx)]
    types = [e.type for e in events]
    assert types.index("tool_call") < types.index("tool_result") < types.index("done")
    tool_result = next(e for e in events if e.type == "tool_result")
    assert "无法计算" in tool_result.data["content"] or "division by zero" in tool_result.data["content"]
    assert events[-1].data["final_text"] == "除零失败，换个算法"


async def test_stream_unknown_tool_feeds_error_to_llm():
    calls = [
        {"name": "no_such_tool", "args": {"x": 1}, "id": "c1", "type": "tool_call"}
    ]
    session, _, ctx = make_session(
        [AIMessage(content="", tool_calls=calls), AIMessage(content="没有这个工具，我直接回答")]
    )
    events = [event async for event in session.stream("用不存在的工具", thread_id="t1", context=ctx)]
    types = [e.type for e in events]
    assert types.index("tool_call") < types.index("tool_result") < types.index("done")
    tool_result = next(e for e in events if e.type == "tool_result")
    assert "not a valid tool" in tool_result.data["content"]
    assert events[-1].data["final_text"] == "没有这个工具，我直接回答"


# ---- 新契约：session 的 context 必选（core 层运行时兜底） ----


async def test_session_invoke_context_none_raises():
    session, _, _ = make_session([AIMessage(content="好")])
    with pytest.raises(ValueError, match="context"):
        await session.invoke("问", thread_id="t1", context=None)  # type: ignore[arg-type]
