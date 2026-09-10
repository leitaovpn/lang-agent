"""agent 层会话入口：repair → compress → 重试续跑 → 结果塑形/事件分类。

core 的 AgentLoop 是纯 graph 薄封装（invoke/stream 与 graph 同形透传）；
本模块承载全部入口前置逻辑与对外语义，对外行为与原 AgentLoop 完全一致：

- invoke(query, *, thread_id, system, context) -> ConversationResult：一次性拿最终结果
- stream(query, *, thread_id, system, context)：异步迭代产出 AgentEvent（token 级流式）

context（AgentContext：llm/tools/summarizer_llm）每次调用显式注入，
由装配层（deps.get_deps）按请求参数构造并缓存。
"""

import asyncio
import copy
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from lang_agent.core.loop import AgentContext, AgentLoop, AgentLoopConfig
from lang_agent.core.loop.compress import (
    estimate_tokens,
    find_round_start,
    render_messages_for_summary,
)
from lang_agent.core.loop.events import (
    EVENT_DONE,
    EVENT_ERROR,
    AgentEvent,
    ConversationResult,
    classify_message_chunk,
    classify_node_update,
    collect_tool_calls,
    messages_after_last_human,
)
from lang_agent.core.loop.repair import repair_state_for_checkpoint
from lang_agent.core.loop.retry import compute_delay, is_retryable
from lang_agent.plugin.graph import merge_update


class ChatSession:
    """ReAct 会话：把 AgentLoop（graph）与入口自愈逻辑组合成完整的多轮对话语义。"""

    def __init__(
        self, *, loop: AgentLoop, config: AgentLoopConfig | None = None
    ) -> None:
        self.loop = loop
        self.config = config or AgentLoopConfig()
        self._locks: dict[str, asyncio.Lock] = {}

    def run_config(self, thread_id: str) -> RunnableConfig:
        """所有 checkpoint 访问共用 agent/thread 隔离键。"""
        return self._run_config(thread_id)

    def _run_config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": "agent:"
                + json.dumps([self.loop.agent_id, thread_id], separators=(",", ":"))
            },
            "recursion_limit": self.config.recursion_limit,
        }

    def _initial_state(self, query: str, system: str | None) -> dict[str, Any]:
        # system 仅在显式提供时写入：不提供则保留该 thread 既有的 system（checkpoint 恢复）
        initial: dict[str, Any] = {
            "messages": [HumanMessage(content=query)],
            "raw_input": query,
        }
        if system:
            initial["system"] = system
        return initial

    async def _repair_checkpoint_state(self, thread_id: str) -> None:
        """invoke/stream 之前的入口修复：修复本地记忆（checkpoint）里的历史。

        上一轮产生的坏段（重复 id、缺失 ToolMessage 等）在 checkpoint 里按原样
        存储；本次调用开始前先修复并写回，避免下次 checkpoint 拉取到错误消息。

        写回方式：add_messages reducer 对同 id 消息原位替换、新消息一律追加到
        末尾——新合成的错误 ToolMessage 会跑到最后，破坏顺序。因此先用
        RemoveMessage 全删再按修复后的顺序加回，精确重建消息列表
        （此时新 HumanMessage 尚未入 state，顺序天然是 [AI, ToolMessage, ...]）。
        """
        config = self._run_config(thread_id)
        state = await self.loop.graph.aget_state(config)
        current: list[BaseMessage] = (state.values or {}).get("messages", [])
        if not current:
            return
        pending_tools = bool(set(state.next) & {"after_model", "before_tool", "tools"})
        if pending_tools:
            cut = next(
                (
                    i
                    for i in range(len(current) - 1, -1, -1)
                    if isinstance((message := current[i]), AIMessage)
                    and message.tool_calls
                ),
                len(current),
            )
            repaired = repair_state_for_checkpoint(list(current[:cut])) + current[cut:]
        else:
            repaired = repair_state_for_checkpoint(list(current))
        if repaired == current:
            return
        await self.loop.graph.aupdate_state(
            config,
            {
                "messages": [
                    RemoveMessage(id=message.id)
                    for message in current
                    if message.id is not None
                ]
                + repaired
            },
        )

    async def _compress_checkpoint_state(
        self, thread_id: str, ctx: AgentContext
    ) -> None:
        """invoke/stream 之前的入口压缩（在 _repair_checkpoint_state 之后执行）。

        token 估算超阈值时：把保留窗口（compress_keep_last）之前的完整轮次交给
        摘要模型总结，摘要写回 checkpoint 的 summary 字段、旧消息用 RemoveMessage
        删除——切点由 find_round_start 保证落在轮起点，tool_call 段永不拆散，
        压缩后历史仍满足 repair 不变式。未超阈值时零成本返回。
        """
        if not self.config.compress_enabled:
            return
        config = self._run_config(thread_id)
        state = await self.loop.graph.aget_state(config)
        current: list[BaseMessage] = list((state.values or {}).get("messages", []))
        if not current:
            return
        tokens = estimate_tokens(current, ctx.llm)
        if tokens <= self.config.compress_token_threshold:
            return
        cut = find_round_start(current, len(current) - self.config.compress_keep_last)
        if cut <= 0:
            return  # 保留窗口已覆盖全部历史，无可压缩
        old = current[:cut]
        rendered = render_messages_for_summary(old)
        prompt = (
            "请把以下对话历史总结成简短摘要，保留：用户目标与偏好、关键结论与事实、"
            "工具调用的重要结果。只输出摘要本身：\n\n" + rendered
        )
        summarizer = ctx.summarizer_llm or ctx.llm
        response = await summarizer.ainvoke([HumanMessage(content=prompt)])
        segment = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        old_summary = (state.values or {}).get("summary") or ""
        new_summary = (old_summary + "\n" if old_summary else "") + segment
        await self.loop.graph.aupdate_state(
            config,
            {
                "summary": new_summary,
                "messages": [RemoveMessage(id=m.id) for m in old if m.id is not None],
            },
        )

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """attempt（已执行次数）后判定是否重试：白名单内且未达上限。"""
        return attempt < self.config.retry_max_attempts and is_retryable(
            exc, self.config.retryable_exceptions
        )

    async def _wait_before_retry(self, exc: Exception, attempt: int) -> None:
        delay = compute_delay(
            attempt,
            self.config.retry_base_delay,
            self.config.retry_backoff_factor,
        )
        logging.getLogger(__name__).warning(
            "graph 调用失败（第 %d 次尝试）：%s: %s；%.1f 秒后重试",
            attempt,
            type(exc).__name__,
            exc,
            delay,
        )
        await asyncio.sleep(delay)

    async def _resume_input(
        self, thread_id: str, config: RunnableConfig, initial: Any
    ) -> Any:
        """重试输入：checkpoint 有 pending 任务 → 先修复 checkpoint 里的消息
        （本轮中途产生的坏段在入口修复之后才写入，续跑前必须修掉，否则 agent
        节点会拿到未修复的历史），再 input=None 从 checkpoint 续跑（失败的超步
        重执行，输入消息不会重复追加）；尚无 checkpoint（首步即失败）→ 复用原输入。"""
        state = await self.loop.graph.aget_state(config)
        if state.next or state.values:
            await self._repair_checkpoint_state(thread_id)
            return None
        return initial

    async def migrate_legacy_thread(self, thread_id: str) -> None:
        """显式复制已完成的旧裸 thread_id 历史；保留源数据供回滚。"""
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        async with lock:
            source = await self.loop.graph.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            target_config = self._run_config(thread_id)
            target = await self.loop.graph.aget_state(target_config)
            if target.values or target.next:
                raise ValueError("目标 agent 会话已存在，不能覆盖")
            if (
                not source.values
                or source.next
                or any(t.interrupts for t in source.tasks)
            ):
                raise ValueError("只能迁移存在且已完成的旧会话")
            if source.values.get("agent_id") not in (None, self.loop.agent_id):
                raise ValueError("源会话属于其他 agent")
            values = copy.deepcopy(source.values)
            values.update(
                agent_id=self.loop.agent_id,
                plugin_revision=self.loop.plugin_revision,
                run_id=uuid4().hex,
            )
            await self.loop.graph.aupdate_state(
                target_config, values, as_node="after_agent"
            )
            migrated = await self.loop.graph.aget_state(target_config)
            if (
                migrated.values.get("messages") != source.values.get("messages")
                or migrated.next
            ):
                raise RuntimeError("迁移校验失败；源历史仍保留")

    def _result(self, thread_id: str, snapshot: Any) -> ConversationResult:
        messages = messages_after_last_human(snapshot.values.get("messages", []))
        interruptions = [
            {"id": item.id, "value": item.value}
            for task in snapshot.tasks
            for item in task.interrupts
        ]
        text = next(
            (
                m.content
                for m in reversed(messages)
                if isinstance(m, AIMessage)
                and isinstance(m.content, str)
                and not m.tool_calls
                and not m.invalid_tool_calls
            ),
            "",
        )
        return ConversationResult(
            thread_id,
            text if not interruptions else "",
            messages,
            collect_tool_calls(messages),
            "interrupted" if interruptions else "completed",
            interruptions,
            self.loop.agent_id,
        )

    async def _drive(
        self,
        query: str | None,
        *,
        thread_id: str,
        system: str | None,
        context: AgentContext,
        answers: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent | ConversationResult]:
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        async with lock:
            config = self._run_config(thread_id)
            snapshot = await self.loop.graph.aget_state(config)
            if query is not None:
                if snapshot.next:
                    raise ValueError("当前会话尚未完成，请先恢复或显式终止")
                ctx = self.loop.bind_plugin_context(context)
                await self._repair_checkpoint_state(thread_id)
                await self._compress_checkpoint_state(thread_id, ctx)
                initial: Any = self._initial_state(query or "", system)
                initial.update(
                    agent_id=self.loop.agent_id,
                    plugin_revision=ctx.plugin_bundle.revision,
                    run_id=uuid4().hex,
                )
            else:
                ids = {item.id for task in snapshot.tasks for item in task.interrupts}
                if ids:
                    if answers is None or set(answers) != ids:
                        raise ValueError("审批答案必须对应当前全部待恢复 interrupt id")
                elif not snapshot.next or answers:
                    raise ValueError("没有可恢复的失败任务或审批 id")
                if snapshot.values.get("agent_id") != self.loop.agent_id:
                    raise ValueError("审批 agent_id 不匹配")
                ctx = self.loop.bind_plugin_context(
                    context, revision=snapshot.values["plugin_revision"]
                )
                if ids and answers is not None:
                    self.loop.validate_plugin_answers(
                        snapshot.values["plugin_revision"],
                        [item for task in snapshot.tasks for item in task.interrupts],
                        answers,
                    )
                initial = Command(resume=answers) if ids else None
                if not ids:
                    await self._repair_checkpoint_state(thread_id)
            snapshot = await self.loop.graph.aget_state(config)
            projection = copy.deepcopy(snapshot.values)
            if isinstance(initial, dict):
                projection = merge_update(projection, initial)
            emitted = set()
            buffered = ctx.plugin_bundle.buffered
            for attempt in range(1, self.config.retry_max_attempts + 1):
                graph_input = (
                    initial
                    if attempt == 1
                    else await self._resume_input(thread_id, config, initial)
                )
                try:
                    async for mode, payload in self.loop.stream(
                        graph_input,
                        config,
                        context=ctx,
                        stream_mode=["messages", "updates"],
                    ):
                        if mode == "messages" and not buffered:
                            chunk, metadata = payload
                            if metadata.get("plugin_model_role") == "primary":
                                event = classify_message_chunk(chunk, metadata)
                                if event:
                                    yield event
                        elif mode == "updates" and isinstance(payload, dict):
                            for node, delta in payload.items():
                                if delta is None:
                                    delta = {}
                                if node == "__interrupt__" or not isinstance(
                                    delta, dict
                                ):
                                    continue
                                projection = merge_update(projection, delta)
                                if node == "before_tool":
                                    message = next(
                                        (
                                            m
                                            for m in reversed(
                                                projection.get("messages", [])
                                            )
                                            if isinstance(m, AIMessage)
                                        ),
                                        None,
                                    )
                                    candidates = classify_node_update(
                                        "agent",
                                        {"messages": [message]} if message else {},
                                    )
                                elif node == "after_tool":
                                    batch: list[BaseMessage] = []
                                    for m in reversed(projection.get("messages", [])):
                                        if isinstance(m, AIMessage):
                                            break
                                        if isinstance(m, BaseMessage):
                                            batch.insert(0, m)
                                    candidates = classify_node_update(
                                        "tools", {"messages": batch}
                                    )
                                else:
                                    candidates = []
                                for event in candidates:
                                    identity = (
                                        event.type,
                                        event.data.get(
                                            "id", event.data.get("tool_call_id")
                                        ),
                                        len(
                                            [
                                                m
                                                for m in projection.get("messages", [])
                                                if isinstance(m, AIMessage)
                                            ]
                                        ),
                                    )
                                    if identity not in emitted:
                                        emitted.add(identity)
                                        yield event
                    break
                except Exception as exc:
                    if not self._should_retry(exc, attempt):
                        raise
                    await self._wait_before_retry(exc, attempt)
                    projection = copy.deepcopy(
                        (await self.loop.graph.aget_state(config)).values
                    )
            result = self._result(thread_id, await self.loop.graph.aget_state(config))
            if buffered and result.status == "completed" and result.final_text:
                yield AgentEvent("llm_token", {"text": result.final_text})
            yield result

    async def invoke(
        self,
        query: str,
        *,
        thread_id: str,
        system: str | None = None,
        context: AgentContext,
    ) -> ConversationResult:
        async for item in self._drive(
            query, thread_id=thread_id, system=system, context=context
        ):
            if isinstance(item, ConversationResult):
                return item
        raise RuntimeError("会话缺少最终结果")

    async def resume(
        self,
        *,
        thread_id: str,
        context: AgentContext,
        answers: dict[str, Any] | None = None,
    ) -> ConversationResult:
        async for item in self._drive(
            None, thread_id=thread_id, system=None, context=context, answers=answers
        ):
            if isinstance(item, ConversationResult):
                return item
        raise RuntimeError("恢复缺少最终结果")

    async def _events(
        self, source: AsyncIterator[AgentEvent | ConversationResult]
    ) -> AsyncIterator[AgentEvent]:
        try:
            async for item in source:
                if isinstance(item, AgentEvent):
                    yield item
                elif item.status == "interrupted":
                    yield AgentEvent(
                        "interrupt",
                        {
                            "agent_id": item.agent_id,
                            "thread_id": item.thread_id,
                            "interrupts": item.interrupts,
                        },
                    )
                else:
                    yield AgentEvent(
                        EVENT_DONE,
                        {
                            "agent_id": item.agent_id,
                            "thread_id": item.thread_id,
                            "final_text": item.final_text,
                            "tool_calls": item.tool_calls,
                        },
                    )
        except Exception as exc:  # noqa: BLE001 统一转为公开错误事件
            yield AgentEvent(EVENT_ERROR, {"message": f"{type(exc).__name__}: {exc}"})

    def stream(
        self,
        query: str,
        *,
        thread_id: str,
        system: str | None = None,
        context: AgentContext,
    ) -> AsyncIterator[AgentEvent]:
        return self._events(
            self._drive(query, thread_id=thread_id, system=system, context=context)
        )

    def resume_stream(
        self,
        *,
        thread_id: str,
        context: AgentContext,
        answers: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        return self._events(
            self._drive(
                None, thread_id=thread_id, system=None, context=context, answers=answers
            )
        )
