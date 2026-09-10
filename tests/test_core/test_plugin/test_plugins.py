"""插件注册、执行、动态更新与审批的行为约束。"""

import asyncio
import threading
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from lang_agent.agent.orchestration.session import ChatSession
from lang_agent.core.loop import AgentContext, AgentLoop
from lang_agent.core.plugin import HookPause, HookResult, PluginBase, PluginRegistry
from tests.conftest import FakeChatModel


class Recorder(PluginBase):
    def __init__(self, name, log, parents=()):
        self.name = name
        self.log = log
        self.parent_plugins = parents

    def before_model(self, state, runtime):
        self.log.append(self.name + ".sync")

    async def abefore_model(self, state, runtime):
        self.log.append(self.name + ".async")


def install(loop, *plugins):
    registry = PluginRegistry()
    for plugin, hooks in plugins:
        registry.register(plugin, agent_id=loop.agent_id, hooks=hooks)
    loop.update_plugin_hooks(registry.snapshot(agent_id=loop.agent_id))
    return registry


def test_registry_isolates_agents_and_checks_dependencies():
    a, b = AgentLoop(), AgentLoop()
    assert a.agent_id and a.agent_id != b.agent_id
    with pytest.raises(AttributeError):
        a.agent_id = b.agent_id
    registry = PluginRegistry()
    registry.register(
        Recorder("parent", []), agent_id=a.agent_id, hooks=["before_model"]
    )
    with pytest.raises(ValueError, match="依赖"):
        registry.register(
            Recorder("child", [], ("parent",)),
            agent_id=b.agent_id,
            hooks=["before_model"],
        )
    assert not registry.snapshot(agent_id=b.agent_id).registrations
    with pytest.raises(ValueError, match="agent"):
        b.update_plugin_hooks(registry.snapshot(agent_id=a.agent_id))
    registry.register(
        Recorder("parent", []), agent_id=b.agent_id, hooks=["before_model"]
    )
    assert len(registry.snapshot(agent_id=b.agent_id).registrations) == 1


async def test_dual_hook_one_node_and_order():
    log = []
    loop = AgentLoop()
    registry = install(
        loop,
        (Recorder("a", log), ["before_model", "abefore_model"]),
        (Recorder("b", log, ("a",)), ["before_model"]),
    )
    context = loop.bind_plugin_context(
        AgentContext(llm=FakeChatModel(responses=[AIMessage(content="好")]), tools=[])
    )
    await loop.invoke(
        {
            "messages": [HumanMessage(content="问")],
            "agent_id": loop.agent_id,
            "plugin_revision": loop.plugin_revision,
            "run_id": uuid4().hex,
        },
        {"configurable": {"thread_id": "x"}},
        context=context,
    )
    assert log == ["a.async", "b.async"]
    with pytest.raises(ValueError, match="依赖"):
        registry.unregister("a", agent_id=loop.agent_id)


async def test_sync_hook_runs_off_loop_and_message_patches_compose():
    main_thread = threading.get_ident()

    class Sync(PluginBase):
        name = "sync"

        def before_model(self, state, runtime):
            assert threading.get_ident() != main_thread
            state["messages"][0].content = "不应泄漏"
            return HookResult(
                update={"messages": [SystemMessage(content="提示", id="prompt")]}
            )

    loop = AgentLoop()
    install(loop, (Sync(), ["before_model"]))
    llm = FakeChatModel(responses=[AIMessage(content="好")])
    result = await ChatSession(loop=loop).invoke(
        "原文", thread_id="x", context=AgentContext(llm=llm, tools=[])
    )
    assert result.messages[0].content == "原文"
    assert sum(m.id == "prompt" for m in result.messages) == 1


async def test_wrap_model_and_async_only_tool():
    log = []

    @tool
    async def double(value: int) -> int:
        """计算两倍。"""
        return value * 2

    class Wrap(PluginBase):
        name = "wrap"

        async def awrap_model_hook(self, request, handler):
            response = await handler(
                request.override(
                    messages=[SystemMessage(content="临时提示"), *request.messages]
                )
            )
            if response.message and response.message.content:
                return response.override(
                    message=response.message.model_copy(update={"content": "最终文本"})
                )
            return response

        def wrap_tool_hook(self, request, handler):
            log.append("request")
            call = {**request.tool_call, "args": {"value": 3}}
            result = handler(request.override(tool_call=call))
            log.append("response")
            return result

    loop = AgentLoop()
    install(loop, (Wrap(), ["wrap_model_hook", "wrap_tool_hook"]))
    llm = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "double", "args": {"value": 1}, "id": "c"}],
            ),
            AIMessage(content="原回答"),
        ]
    )
    events = [
        e
        async for e in ChatSession(loop=loop).stream(
            "问", thread_id="x", context=AgentContext(llm=llm, tools=[double])
        )
    ]
    assert events[-1].data["final_text"] == "最终文本"
    assert next(e for e in events if e.type == "tool_result").data["content"] == "6"
    assert log == ["request", "response"]
    assert llm.seen_messages[0][0].content == "临时提示"


async def test_pause_update_resume_old_revision_and_next_round_new_revision():
    log = []

    class Approval(PluginBase):
        name = "approval"

        def before_tool(self, state, runtime):
            if "approve" not in runtime.answers:
                return HookResult(
                    pause=HookPause(key="approve", payload={"question": "继续？"})
                )
            assert runtime.answers["approve"] is True

    @tool
    async def work() -> str:
        """执行一次工作。"""
        log.append("tool")
        return "结果"

    loop = AgentLoop()
    graph = loop.graph
    install(
        loop, (Recorder("old", log), ["before_model"]), (Approval(), ["before_tool"])
    )
    old = loop.plugin_revision
    llm = FakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "work", "args": {}, "id": "c"}]),
            AIMessage(content="完成"),
            AIMessage(content="新轮"),
        ]
    )
    context = AgentContext(llm=llm, tools=[work])
    session = ChatSession(loop=loop)
    paused = await session.invoke("问", thread_id="x", context=context)
    assert paused.status == "interrupted"
    assert "tool" not in log
    with pytest.raises(ValueError, match="恢复"):
        await session.invoke("不能追加", thread_id="x", context=context)
    install(loop, (Recorder("new", log), ["before_model"]))
    assert loop.graph is graph and loop.plugin_revision != old
    answer = {paused.interrupts[0]["id"]: True}
    result = await session.resume(thread_id="x", answers=answer, context=context)
    assert result.status == "completed"
    assert log == ["old.async", "tool", "old.async"]
    await session.invoke("下轮", thread_id="x", context=context)
    assert log[-1] == "new.async"
    config = session.run_config("x")
    checkpoints = [c async for c in loop.graph.checkpointer.alist(config)]
    assert {c.config["configurable"]["checkpoint_ns"] for c in checkpoints} == {""}


async def test_shared_saver_same_thread_isolated():
    saver = InMemorySaver()
    a, b = (
        ChatSession(loop=AgentLoop(checkpointer=saver)),
        ChatSession(loop=AgentLoop(checkpointer=saver)),
    )
    for session in (a, b):
        llm = FakeChatModel(responses=[AIMessage(content="答")])
        await session.invoke(
            "问", thread_id="same", context=AgentContext(llm=llm, tools=[])
        )
        assert len(llm.seen_messages[0]) == 1


async def test_update_during_run_keeps_snapshot():
    started, release = asyncio.Event(), asyncio.Event()
    seen = []

    class Blocking(PluginBase):
        name = "blocking"

        async def abefore_model(self, state, runtime):
            started.set()
            await release.wait()

        def after_agent(self, state, runtime):
            seen.append("old")

    loop = AgentLoop()
    install(loop, (Blocking(), ["before_model", "after_agent"]))
    task = asyncio.create_task(
        ChatSession(loop=loop).invoke(
            "问",
            thread_id="x",
            context=AgentContext(
                llm=FakeChatModel(responses=[AIMessage(content="答")]), tools=[]
            ),
        )
    )
    await started.wait()
    install(loop)
    release.set()
    await task
    assert seen == ["old"]


async def test_sqlite_approval_survives_restart(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from lang_agent.core.plugin.approval import ToolApprovalPlugin

    path = str(tmp_path / "approval.sqlite")
    executed = []

    @tool
    async def work(value: int) -> str:
        """记录执行参数。"""
        executed.append(value)
        return str(value)

    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        loop = AgentLoop(checkpointer=saver)
        install(loop, (ToolApprovalPlugin(), ["before_tool"]))
        agent_id = loop.agent_id
        session = ChatSession(loop=loop)
        llm = FakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "work", "args": {"value": 1}, "id": "c"}],
                )
            ]
        )
        paused = await session.invoke(
            "执行", thread_id="t", context=AgentContext(llm=llm, tools=[work])
        )
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        loop = AgentLoop.restore(agent_id=agent_id, checkpointer=saver)
        install(loop, (ToolApprovalPlugin(), ["before_tool"]))
        result = await ChatSession(loop=loop).resume(
            thread_id="t",
            answers={
                paused.interrupts[0]["id"]: {"action": "edit", "args": {"value": 5}}
            },
            context=AgentContext(
                llm=FakeChatModel(responses=[AIMessage(content="完成")]), tools=[work]
            ),
        )
        assert result.final_text == "完成"
        assert executed == [5]


@pytest.mark.parametrize("approval_name", ["tool_approval", "custom_approval"])
async def test_approval_reject_never_enters_wrapper(approval_name):
    from lang_agent.core.plugin.approval import ToolApprovalPlugin

    calls = []

    class Wrap(PluginBase):
        name = "wrap"

        def wrap_tool_hook(self, request, handler):
            calls.append("wrapper")
            return handler(request)

    approval = ToolApprovalPlugin()
    approval.name = approval_name
    loop = AgentLoop()
    install(loop, (approval, ["before_tool"]), (Wrap(), ["wrap_tool_hook"]))
    ctx = AgentContext(
        llm=FakeChatModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[{"name": "x", "args": {}, "id": "c"}]
                ),
                AIMessage(content="已拒绝"),
            ]
        ),
        tools=[],
    )
    session = ChatSession(loop=loop)
    paused = await session.invoke("问", thread_id="t", context=ctx)
    result = await session.resume(
        thread_id="t", answers={paused.interrupts[0]["id"]: "reject"}, context=ctx
    )
    assert result.final_text == "已拒绝"
    assert calls == []


def test_enforce_approval_follows_registered_name_and_rejects_ambiguity():
    from types import SimpleNamespace

    from lang_agent.core.plugin.approval import enforce_approval

    request = SimpleNamespace(tool_call={"id": "c", "name": "x", "args": {}})

    def state(namespaces):
        return {"run_id": "r1", "plugin_state": namespaces}

    # 决定存放在自定义注册名之下，同样生效。
    denial = enforce_approval(
        state(
            {
                "custom_approval": {
                    "run_id": "r1",
                    "decisions": {"c": {"action": "reject", "fingerprint": "f"}},
                }
            }
        ),
        request,
        plugin_names=("custom_approval",),
    )
    assert denial is not None and denial.status == "error"
    # 旧轮或无决定时不拦截。
    assert (
        enforce_approval(
            state({"custom_approval": {"run_id": "old", "decisions": {}}}),
            request,
            plugin_names=("custom_approval",),
        )
        is None
    )
    assert (
        enforce_approval(state({}), request, plugin_names=("custom_approval",)) is None
    )
    # 同一轮多个审批来源属于配置错误。
    with pytest.raises(ValueError, match="审批"):
        enforce_approval(
            state(
                {
                    "a": {"run_id": "r1", "decisions": {"c": {"action": "reject"}}},
                    "b": {"run_id": "r1", "decisions": {"c": {"action": "approve"}}},
                }
            ),
            request,
            plugin_names=("a", "b"),
        )


async def test_buffered_after_hook_replaces_final_without_leaking_tokens():
    class Review(PluginBase):
        name = "review"
        requires_buffered_output = True

        def after_agent(self, state, runtime):
            m = state["messages"][-1].model_copy(update={"content": "审核后"})
            return HookResult(update={"messages": [m]})

    loop = AgentLoop()
    install(loop, (Review(), ["after_agent"]))
    ctx = AgentContext(
        llm=FakeChatModel(responses=[AIMessage(content="未审核原文")]), tools=[]
    )
    events = [
        e async for e in ChatSession(loop=loop).stream("问", thread_id="t", context=ctx)
    ]
    assert "".join(e.data["text"] for e in events if e.type == "llm_token") == "审核后"
    assert events[-1].data["final_text"] == "审核后"


async def test_message_add_replace_remove_is_one_committed_delta():
    from langchain_core.messages import RemoveMessage

    class Add(PluginBase):
        name = "add"

        def before_model(self, state, runtime):
            return HookResult(
                update={"messages": [SystemMessage(content="临时", id="tmp")]}
            )

    class Remove(PluginBase):
        name = "remove"
        parent_plugins = ("add",)

        def before_model(self, state, runtime):
            assert state["messages"][-1].content == "临时"
            return HookResult(update={"messages": [RemoveMessage(id="tmp")]})

    loop = AgentLoop()
    install(loop, (Add(), ["before_model"]), (Remove(), ["before_model"]))
    ctx = AgentContext(llm=FakeChatModel(responses=[AIMessage(content="答")]), tools=[])
    result = await ChatSession(loop=loop).invoke("问", thread_id="t", context=ctx)
    assert [m.content for m in result.messages] == ["问", "答"]


async def test_tool_wrapper_contract_error_fails_before_next_model():
    from langchain_core.messages import ToolMessage

    class Bad(PluginBase):
        name = "bad"

        def wrap_tool_hook(self, request, handler):
            return ToolMessage(content="错配", tool_call_id="wrong")

    loop = AgentLoop()
    install(loop, (Bad(), ["wrap_tool_hook"]))
    ctx = AgentContext(
        llm=FakeChatModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c"}])
            ]
        ),
        tools=[],
    )
    with pytest.raises(ValueError, match="tool_call_id"):
        await ChatSession(loop=loop).invoke("问", thread_id="t", context=ctx)


def test_update_reuses_unchanged_hook_and_rolls_back_conflict():
    from lang_agent.core.plugin.runtime import PluginRuntime

    runtime = PluginRuntime("a")
    registry = PluginRegistry()
    registry.register(Recorder("r", []), agent_id="a", hooks=["before_model"])
    runtime.update(registry.snapshot(agent_id="a"))
    old = runtime.get()

    class After(PluginBase):
        name = "after"

        def after_agent(self, state, runtime):
            pass

    registry.register(After(), agent_id="a", hooks=["after_agent"])
    snapshot = registry.snapshot(agent_id="a")
    with pytest.raises(ValueError, match="冲突"):
        runtime.update(snapshot, "wrong")
    assert runtime.get() is old
    runtime.update(snapshot)
    assert runtime.get().graphs["before_model"] is old.graphs["before_model"]


async def test_multiple_approvals_replay_answers_without_repeating_tools():
    from lang_agent.core.plugin.approval import ToolApprovalPlugin

    called = []

    @tool
    async def work(value: int) -> str:
        """执行工作。"""
        called.append(value)
        return str(value)

    loop = AgentLoop()
    install(loop, (ToolApprovalPlugin(), ["before_tool"]))
    llm = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "work", "args": {"value": 1}, "id": "a"},
                    {"name": "work", "args": {"value": 2}, "id": "b"},
                ],
            ),
            AIMessage(content="完成"),
        ]
    )
    ctx = AgentContext(llm=llm, tools=[work])
    session = ChatSession(loop=loop)
    first = await session.invoke("问", thread_id="t", context=ctx)
    second = await session.resume(
        thread_id="t", answers={first.interrupts[0]["id"]: "approve"}, context=ctx
    )
    assert second.status == "interrupted" and called == []
    result = await session.resume(
        thread_id="t", answers={second.interrupts[0]["id"]: "reject"}, context=ctx
    )
    assert result.status == "completed" and called == [1]


def test_registry_rejects_bad_methods_and_cycle_transactionally():
    registry = PluginRegistry()

    class Empty(PluginBase):
        name = "empty"

    with pytest.raises(ValueError, match="未实现"):
        registry.register(Empty(), agent_id="a", hooks=["before_model"])

    class Bad(PluginBase):
        name = "bad"

        def before_model(self):
            pass

    with pytest.raises(ValueError, match="签名"):
        registry.register(Bad(), agent_id="a", hooks=["before_model"])
    registry.register(Recorder("p", []), agent_id="a", hooks=["before_model"])
    registry.register(Recorder("c", [], ("p",)), agent_id="a", hooks=["before_model"])
    old = registry.snapshot(agent_id="a")
    with pytest.raises(ValueError, match="成环"):
        registry.replace(
            "p", Recorder("p", [], ("c",)), agent_id="a", hooks=["before_model"]
        )
    assert registry.snapshot(agent_id="a") == old


async def test_config_snapshot_not_affected_by_caller_mutation():
    class Configured(PluginBase):
        name = "configured"

        def after_agent(self, state, runtime):
            return HookResult(
                update={
                    "messages": [
                        state["messages"][-1].model_copy(
                            update={"content": self.config["answer"]}
                        )
                    ]
                }
            )

    plugin = Configured()
    plugin.config = {"answer": "旧配置"}
    loop = AgentLoop()
    install(loop, (plugin, ["after_agent"]))
    plugin.config["answer"] = "被外部修改"
    result = await ChatSession(loop=loop).invoke(
        "问",
        thread_id="t",
        context=AgentContext(
            llm=FakeChatModel(responses=[AIMessage(content="原文")]), tools=[]
        ),
    )
    assert result.final_text == "旧配置"


async def test_migrate_completed_legacy_history_preserves_source():
    loop = AgentLoop()
    source = {"configurable": {"thread_id": "legacy"}}
    await loop.invoke(
        {"messages": [HumanMessage(content="旧问题")]},
        source,
        context=AgentContext(
            llm=FakeChatModel(responses=[AIMessage(content="旧答案")]), tools=[]
        ),
    )
    session = ChatSession(loop=loop)
    await session.migrate_legacy_thread("legacy")
    before = await loop.graph.aget_state(source)
    after = await loop.graph.aget_state(session.run_config("legacy"))
    assert before.values["messages"] == after.values["messages"]
    assert after.values["agent_id"] == loop.agent_id
    assert not after.next
    with pytest.raises(ValueError, match="已存在"):
        await session.migrate_legacy_thread("legacy")


async def test_resume_failed_run_without_new_human_message():
    loop = AgentLoop()
    session = ChatSession(loop=loop)
    ctx = AgentContext(llm=FakeChatModel(responses=[]), tools=[])
    with pytest.raises(AssertionError):
        await session.invoke("问", thread_id="t", context=ctx)
    ctx.llm.responses.append(AIMessage(content="恢复成功"))
    result = await session.resume(thread_id="t", context=ctx)
    assert result.final_text == "恢复成功"
    assert len([m for m in result.messages if isinstance(m, HumanMessage)]) == 1


async def test_before_agent_interrupt_can_resume_and_missing_revision_fails():
    from lang_agent.core.plugin import PluginRevisionUnavailable

    class Pause(PluginBase):
        name = "pause"

        def before_agent(self, state, runtime):
            if "start" not in runtime.answers:
                return HookResult(pause=HookPause(key="start", payload="开始？"))

    saver = InMemorySaver()
    loop = AgentLoop(checkpointer=saver)
    install(loop, (Pause(), ["before_agent"]))
    ctx = AgentContext(llm=FakeChatModel(responses=[AIMessage(content="答")]), tools=[])
    paused = await ChatSession(loop=loop).invoke("问", thread_id="t", context=ctx)
    restored = AgentLoop.restore(agent_id=loop.agent_id, checkpointer=saver)
    session = ChatSession(loop=restored)
    with pytest.raises(PluginRevisionUnavailable):
        await session.resume(
            thread_id="t", answers={paused.interrupts[0]["id"]: True}, context=ctx
        )
    install(restored, (Pause(), ["before_agent"]))
    result = await session.resume(
        thread_id="t", answers={paused.interrupts[0]["id"]: True}, context=ctx
    )
    assert result.final_text == "答"


async def test_async_cancellation_releases_session_lock():
    entered = asyncio.Event()

    class Wait(PluginBase):
        name = "wait"

        async def abefore_model(self, state, runtime):
            entered.set()
            await asyncio.Event().wait()

    loop = AgentLoop()
    install(loop, (Wait(), ["before_model"]))
    session = ChatSession(loop=loop)
    task = asyncio.create_task(
        session.invoke(
            "问",
            thread_id="t",
            context=AgentContext(llm=FakeChatModel(responses=[]), tools=[]),
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not session._locks["t"].locked()


async def test_model_wrapper_cache_skips_model_and_nested_order():
    from lang_agent.core.plugin import ModelResponse

    order = []

    class Outer(PluginBase):
        name = "outer"

        async def awrap_model_hook(self, request, handler):
            order.append("outer.request")
            response = await handler(request)
            order.append("outer.response")
            return response

    class Cache(PluginBase):
        name = "cache"
        parent_plugins = ("outer",)

        async def awrap_model_hook(self, request, handler):
            order.append("cache")
            return ModelResponse(AIMessage(content="缓存回答"))

    loop = AgentLoop()
    install(loop, (Outer(), ["wrap_model_hook"]), (Cache(), ["wrap_model_hook"]))
    model = FakeChatModel(responses=[])
    result = await ChatSession(loop=loop).invoke(
        "问", thread_id="t", context=AgentContext(llm=model, tools=[])
    )
    assert result.final_text == "缓存回答" and model.seen_messages == []
    assert order == ["outer.request", "cache", "outer.response"]


async def test_invalid_approval_answer_does_not_consume_interrupt():
    from lang_agent.core.plugin.approval import ToolApprovalPlugin

    loop = AgentLoop()
    install(loop, (ToolApprovalPlugin(), ["before_tool"]))
    session = ChatSession(loop=loop)
    ctx = AgentContext(
        llm=FakeChatModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[{"name": "x", "args": {}, "id": "c"}]
                ),
                AIMessage(content="已拒绝"),
            ]
        ),
        tools=[],
    )
    paused = await session.invoke("问", thread_id="t", context=ctx)
    with pytest.raises(ValueError, match="审批答案"):
        await session.resume(
            thread_id="t", answers={paused.interrupts[0]["id"]: "typo"}, context=ctx
        )
    result = await session.resume(
        thread_id="t", answers={paused.interrupts[0]["id"]: "reject"}, context=ctx
    )
    assert result.status == "completed"


def test_runnable_callable_dispatches_sync_and_async_once():
    from langgraph.runtime import Runtime

    from lang_agent.core.plugin.graph import HookInvocation, build_hook_node

    log = []
    registry = PluginRegistry()
    registry.register(
        Recorder("r", log), agent_id="a", hooks=["before_model", "abefore_model"]
    )
    snapshot = registry.snapshot(agent_id="a")
    node = build_hook_node(snapshot.registrations[0], "before_model")
    runtime = Runtime(
        context=HookInvocation(
            None, {}, "before_model", "a", snapshot.revision, "run", {}
        )
    )
    state = {"working": {"messages": []}, "pause": None, "route": None}
    node.invoke(state, runtime=runtime)
    asyncio.run(node.ainvoke(state, runtime=runtime))
    assert log == ["r.sync", "r.async"]


async def test_empty_cached_response_after_tool_does_not_repeat_tool():
    from lang_agent.core.plugin import ModelResponse

    calls = []

    @tool
    async def work() -> str:
        """执行一次。"""
        calls.append("tool")
        return "结果"

    class Empty(PluginBase):
        name = "empty"

        async def awrap_model_hook(self, request, handler):
            from langchain_core.messages import ToolMessage

            if isinstance(request.messages[-1], ToolMessage):
                return ModelResponse(None)
            return await handler(request)

    loop = AgentLoop()
    install(loop, (Empty(), ["wrap_model_hook"]))
    ctx = AgentContext(
        llm=FakeChatModel(
            responses=[
                AIMessage(
                    content="", tool_calls=[{"name": "work", "args": {}, "id": "c"}]
                )
            ]
        ),
        tools=[work],
    )
    result = await ChatSession(loop=loop).invoke("问", thread_id="t", context=ctx)
    assert result.final_text == "" and calls == ["tool"]
