"""agent 层会话入口：repair → compress → 重试续跑 → 结果塑形/事件分类。

core 的 AgentLoop 是纯 graph 薄封装（invoke/stream 与 graph 同形透传）；
本模块承载全部入口前置逻辑与对外语义，对外行为与原 AgentLoop 完全一致：

- invoke(query, *, thread_id, system, context) -> ConversationResult：一次性拿最终结果
- stream(query, *, thread_id, system, context)：异步迭代产出 AgentEvent（token 级流式）

context（AgentContext：llm/tools/summarizer_llm）每次调用显式注入，
由装配层（deps.get_deps）按请求参数构造并缓存。
"""
import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from lang_agent.core.loop import AgentContext, AgentLoop, AgentLoopConfig
from lang_agent.core.loop.approval import (
    ToolApprovalDecision,
    ToolApprovalRequest,
    validate_approval_decision,
)
from lang_agent.core.loop.compress import (
    estimate_tokens,
    find_round_start,
    render_messages_for_summary,
)
from lang_agent.core.loop.events import (
    EVENT_APPROVAL_REQUIRED,
    EVENT_DONE,
    EVENT_ERROR,
    AgentEvent,
    ConversationResult,
    classify_message_chunk,
    classify_node_update,
    collect_tool_calls,
    messages_after_last_human,
)
from lang_agent.core.loop.react_agent import AGENT_NODE
from lang_agent.core.loop.repair import repair_state_for_checkpoint
from lang_agent.core.loop.retry import compute_delay, is_retryable


class ApprovalPendingError(RuntimeError):
    """线程已有等待人工处理的工具审批。"""


class ApprovalNotFoundError(RuntimeError):
    """线程当前没有可恢复的审批。"""


class ChatSession:
    """ReAct 会话：把 AgentLoop（graph）与入口自愈逻辑组合成完整的多轮对话语义。"""

    def __init__(self, *, loop: AgentLoop, config: AgentLoopConfig | None = None) -> None:
        self.loop = loop
        self.config = config or AgentLoopConfig()

    def _run_config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
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

    async def get_pending_approval(self, thread_id: str) -> dict[str, Any] | None:
        """读取线程当前待审批批次；审批数据来自 checkpoint 的 interrupt。"""
        state = await self.loop.graph.aget_state(self._run_config(thread_id))
        for pending in state.interrupts:
            if isinstance(pending.value, dict) and "approval_id" in pending.value:
                return {**pending.value, "interrupt_id": pending.id}
        return None

    def _result_from_state(
        self,
        state: dict[str, Any],
        *,
        thread_id: str,
        approval: dict[str, Any] | None = None,
    ) -> ConversationResult:
        """把 graph 状态塑形成当前轮结果。"""
        round_messages = messages_after_last_human(state.get("messages", []))
        final_text = ""
        for message in reversed(round_messages):
            if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
                final_text = message.content
                break
        return ConversationResult(
            thread_id=thread_id,
            final_text=final_text,
            messages=round_messages,
            tool_calls=collect_tool_calls(round_messages),
            status="awaiting_approval" if approval else "completed",
            approval=approval,
        )

    def _validate_resume_decision(
        self, pending: dict[str, Any], decision: dict[str, Any]
    ) -> None:
        """在恢复 graph 前校验决定，错误输入不会消耗 checkpoint 的 interrupt。"""
        request = cast(
            ToolApprovalRequest,
            {
                "approval_id": pending["approval_id"],
                "tool_calls": pending["tool_calls"],
            },
        )
        validate_approval_decision(request, cast(ToolApprovalDecision, decision))

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

    async def _compress_checkpoint_state(self, thread_id: str, ctx: AgentContext) -> None:
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
            response.content if isinstance(response.content, str) else str(response.content)
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

    async def invoke(
        self,
        query: str,
        *,
        thread_id: str,
        system: str | None = None,
        context: AgentContext,
    ) -> ConversationResult:
        """同步语义的一次调用：返回最终文本、本轮消息与工具调用汇总。

        messages 与 tool_calls 都只含本轮（最后一条 HumanMessage 起）产生的内容，
        不含该 thread 的历史轮次。context 携带本次 run 的 LLM/工具集（每次注入）。
        """
        pending = await self.get_pending_approval(thread_id)
        if pending is not None:
            raise ApprovalPendingError("当前线程有待审批的工具调用，请先提交审批决定")
        ctx = context
        await self._repair_checkpoint_state(thread_id)
        await self._compress_checkpoint_state(thread_id, ctx)
        initial = self._initial_state(query, system)
        config = self._run_config(thread_id)
        attempt = 0
        while True:
            attempt += 1
            graph_input = (
                initial if attempt == 1 else await self._resume_input(thread_id, config, initial)
            )
            try:
                state = await self.loop.invoke(graph_input, config, context=ctx)
                break
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._wait_before_retry(exc, attempt)
        approval = None
        interrupts = state.get("__interrupt__", [])
        if interrupts:
            value = interrupts[0].value
            if isinstance(value, dict):
                approval = {**value, "interrupt_id": interrupts[0].id}
        return self._result_from_state(state, thread_id=thread_id, approval=approval)

    async def resume(
        self,
        decision: dict[str, Any],
        *,
        thread_id: str,
        context: AgentContext,
    ) -> ConversationResult:
        """提交人工审批决定，从 checkpoint 恢复同一次 graph 运行。"""
        pending = await self.get_pending_approval(thread_id)
        if pending is None:
            raise ApprovalNotFoundError("当前线程没有待审批的工具调用")
        self._validate_resume_decision(pending, decision)
        config = self._run_config(thread_id)
        initial: Any = Command(resume=decision)
        attempt = 0
        while True:
            attempt += 1
            graph_input = (
                initial if attempt == 1 else await self._resume_input(thread_id, config, initial)
            )
            try:
                state = await self.loop.invoke(graph_input, config, context=context)
                break
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._wait_before_retry(exc, attempt)
        next_approval = None
        interrupts = state.get("__interrupt__", [])
        if interrupts:
            value = interrupts[0].value
            if isinstance(value, dict):
                next_approval = {**value, "interrupt_id": interrupts[0].id}
        return self._result_from_state(
            state, thread_id=thread_id, approval=next_approval
        )

    async def stream(
        self,
        query: str,
        *,
        thread_id: str,
        system: str | None = None,
        context: AgentContext,
    ) -> AsyncIterator[AgentEvent]:
        """流式执行：依次产出 llm_token / tool_call / tool_result，最后 done 或 error。

        基于 graph.astream(stream_mode=["messages", "updates"])：
        - messages 通道给 token 级文本（只取 agent 节点的 chunk）
        - updates 通道给完整 AIMessage（tool_call 决策）与 ToolMessage（tool_result）
        context 携带本次 run 的 LLM/工具集（每次注入，重试续跑沿用同一份）。
        """
        try:
            pending = await self.get_pending_approval(thread_id)
            if pending is not None:
                raise ApprovalPendingError("当前线程有待审批的工具调用，请先提交审批决定")
            await self._repair_checkpoint_state(thread_id)
            initial = self._initial_state(query, system)
            async for event in self._stream_graph(
                initial, thread_id=thread_id, context=context
            ):
                yield event
        except Exception as exc:  # noqa: BLE001 统一转 error 事件，不外泄 traceback
            yield AgentEvent(EVENT_ERROR, {"message": f"{type(exc).__name__}: {exc}"})

    async def resume_stream(
        self,
        decision: dict[str, Any],
        *,
        thread_id: str,
        context: AgentContext,
    ) -> AsyncIterator[AgentEvent]:
        """流式提交审批决定并继续原 graph 运行。"""
        try:
            pending = await self.get_pending_approval(thread_id)
            if pending is None:
                raise ApprovalNotFoundError("当前线程没有待审批的工具调用")
            self._validate_resume_decision(pending, decision)
            async for event in self._stream_graph(
                Command(resume=decision), thread_id=thread_id, context=context
            ):
                yield event
        except Exception as exc:  # noqa: BLE001 统一转 error 事件，不外泄 traceback
            yield AgentEvent(EVENT_ERROR, {"message": f"{type(exc).__name__}: {exc}"})

    async def _stream_graph(
        self,
        initial: Any,
        *,
        thread_id: str,
        context: AgentContext,
    ) -> AsyncIterator[AgentEvent]:
        """流式驱动新输入或 Command(resume=...)，并统一分类 graph 事件。"""
        ctx = context
        final_text = ""
        config = self._run_config(thread_id)
        attempt = 0
        while True:
            attempt += 1
            graph_input = (
                initial if attempt == 1 else await self._resume_input(thread_id, config, initial)
            )
            interrupted = False
            try:
                async for mode, payload in self.loop.stream(
                    graph_input,
                    config,
                    context=ctx,
                    stream_mode=["messages", "updates"],
                ):
                    if mode == "messages" and isinstance(payload, tuple) and len(payload) == 2:
                        chunk, metadata = payload
                        if isinstance(chunk, BaseMessage) and isinstance(metadata, dict):
                            event = classify_message_chunk(chunk, metadata)
                            if event:
                                yield event
                    elif mode == "updates" and isinstance(payload, dict):
                        for node, delta in payload.items():
                            if node == "__interrupt__":
                                for item in delta:
                                    if isinstance(item.value, dict):
                                        yield AgentEvent(
                                            EVENT_APPROVAL_REQUIRED,
                                            {**item.value, "interrupt_id": item.id},
                                        )
                                        interrupted = True
                                continue
                            if node == AGENT_NODE:
                                for message in delta.get("messages", []):
                                    if (
                                        isinstance(message, AIMessage)
                                        and isinstance(message.content, str)
                                        and message.content
                                        and not message.tool_calls
                                        and not message.invalid_tool_calls
                                    ):
                                        final_text = message.content
                            for event in classify_node_update(node, delta):
                                yield event
                break
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._wait_before_retry(exc, attempt)
        if interrupted:
            return
        state = await self.loop.graph.aget_state(config)
        round_messages = messages_after_last_human(state.values.get("messages", []))
        tool_calls = collect_tool_calls(round_messages)
        yield AgentEvent(
            EVENT_DONE,
            {
                "thread_id": thread_id,
                "final_text": final_text,
                "tool_calls": tool_calls,
            },
        )
