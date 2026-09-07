"""core 层：手搓 StateGraph 的完整 ReAct agent loop。

拓扑：
    START → agent（LLM + bind_tools，流式合并）
              ├─ 无 tool_calls → END
              ├─ 只有 invalid_tool_calls（已附错误反馈）→ agent（重试）
              └─ 有 tool_calls → tools（ToolNode 执行真实工具）→ agent（回环）

invoke/stream 入口先做 _repair_checkpoint_state：修复并写回 checkpoint 里的历史，
避免下次拉取到错误消息。

用 checkpointer 按 thread_id 记忆多轮对话；对外统一暴露 invoke / stream，
屏蔽底层 agent（图结构、工具、LLM）差异。

注意：本模块不能使用 `from __future__ import annotations`——langgraph 会用本模块的
globals 求值 AgentState 的 `Annotated[...]` 注解，字符串化会导致 NameError。
"""
import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict
from langgraph.runtime import Runtime
from lang_agent.core.events import (
    EVENT_DONE,
    EVENT_ERROR,
    AgentEvent,
    ConversationResult,
    classify_message_chunk,
    classify_node_update,
    collect_tool_calls,
    messages_after_last_human,
)
from lang_agent.core.repair import INVALID_ID_PREFIX, repair_state_for_checkpoint
from lang_agent.core.retry import compute_delay, is_retryable
from lang_agent.core.tool_registry import instantiate_tools

AGENT_NODE = "agent"
TOOLS_NODE = "tools"


class AgentState(TypedDict):
    """agent 循环状态：messages 用 add_messages reducer 累加；system 为每线程静态元数据。"""

    messages: Annotated[list[AnyMessage], add_messages]
    system: str
    raw_input: str


def should_continue(state: AgentState) -> str:
    """循环条件：以最近一条 AIMessage 决定走向（工具结果/错误反馈由固定边处理）。

    - 有合法 tool_calls → tools（执行工具）
    - 只有 invalid_tool_calls（agent 节点已附错误反馈 ToolMessage）→ agent 重试
    - 纯文本回答 → END
    """
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            if message.tool_calls:
                return TOOLS_NODE
            if message.invalid_tool_calls:
                return AGENT_NODE
            return END
    return END


class AgentContext(BaseModel):
    """每次 run 的静态上下文：LLM 与工具集。

    langgraph 0.6 的 context 特性：不写入 checkpoint、run 内只读、共享给所有节点。
    节点通过注入的 runtime 对象访问（runtime.context）——0.6.11 不支持以
    `context` 参数名直接注入节点函数。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: BaseChatModel
    tools: list[BaseTool]


@dataclass
class AgentLoopConfig:
    checkpointer_kind: str = "memory"  # "memory" | "sqlite"
    db_path: str = "~/.lang-agent/checkpoints.sqlite"
    recursion_limit: int = 25
    # graph 调用异常重试（瞬时异常白名单 + 指数退避）
    retry_max_attempts: int = 3  # 总尝试次数（含首次）
    retry_base_delay: float = 0.5  # 首次退避秒数
    retry_backoff_factor: float = 2.0  # 指数退避因子
    retryable_exceptions: Optional[tuple] = None  # None → 默认白名单（见 core/retry.py）


async def build_checkpointer(config: AgentLoopConfig):
    """按配置构建 checkpointer（异步工厂）。

    sqlite 用 AsyncSqliteSaver，其构造绑定当前事件循环，必须在异步上下文中创建；
    memory 用 InMemorySaver，可在任意位置创建。
    """
    if config.checkpointer_kind == "sqlite":
        import aiosqlite

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        if config.db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(config.db_path))
            os.makedirs(parent, exist_ok=True)
        conn = await aiosqlite.connect(config.db_path)
        return AsyncSqliteSaver(conn)
    if config.checkpointer_kind == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    raise ValueError(f"未知 checkpointer_kind: {config.checkpointer_kind!r}")


class AgentLoop:
    """ReAct agent 循环统一入口。

    - invoke(query, *, thread_id, system) -> ConversationResult：一次性拿最终结果
    - stream(query, *, thread_id, system)：异步迭代产出 AgentEvent（token 级流式）
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[list[BaseTool]] = None,
        config: Optional[AgentLoopConfig] = None,
        checkpointer=None,
    ):
        self._llm = llm
        self._config = config or AgentLoopConfig()
        if checkpointer is not None:
            self._checkpointer = checkpointer
        elif self._config.checkpointer_kind == "memory":
            from langgraph.checkpoint.memory import InMemorySaver

            self._checkpointer = InMemorySaver()
        else:
            raise ValueError(
                "sqlite checkpointer 必须在异步上下文中创建："
                "请先 await build_checkpointer(config) 再注入 checkpointer 参数"
            )
        self._tools = self._build_tools(tools)
        self._context = AgentContext(llm=self._llm, tools=self._tools)
        self._graph = self._build_graph()

    def _build_tools(self, tools:  Optional[list[BaseTool]] = None) -> list[BaseTool]:
        """按配置构建工具列表（可被子类覆盖）。"""
        default_tools = instantiate_tools()
        return default_tools + (tools or [])

    def _build_graph(self):
        async def agent_node(
            state: AgentState, config: RunnableConfig, runtime: Runtime
        ) -> dict[str, list[BaseMessage]]:
            # agent 节点：流式调用 LLM 并合并 chunk，保证 token 级事件可被捕获。
            # 必须把 config 传给 astream：否则 bind_tools 的 RunnableBinding 内层
            # 模型收不到回调，token 级流式事件会丢失（langgraph 0.6 行为）。
            # LLM 与工具集从 runtime.context 取（每次 run 注入，见 AgentContext）。
            # 历史在 invoke/stream 入口已修复并写回 checkpoint，这里原样使用。
            context = cast(AgentContext, runtime.context)
            model = context.llm.bind_tools(context.tools)
            messages: list[BaseMessage] = list(state["messages"])
            if state.get("system"):
                messages = [SystemMessage(content=state["system"])] + messages
            chunks: list[BaseMessageChunk] = []
            async for chunk in model.astream(messages, config=config):
                chunks.append(cast(BaseMessageChunk, chunk))
            if not chunks:
                return {"messages": []}
            merged = cast(AIMessageChunk, chunks[0])
            for chunk in chunks[1:]:
                merged = merged + cast(AIMessageChunk, chunk)
            result: list[BaseMessage] = [merged]  # type: ignore
            if not merged.tool_calls:  # type: ignore
                # 无合法调用：为解析失败的调用附错误反馈 ToolMessage，驱动循环重试。
                # id 与 repair 层的确定性 id 一致（invalid_<消息下标>_<条内序号>），
                # 避免下一轮修复时重复合成。
                base_index = len(state["messages"])
                for k, invalid in enumerate(merged.invalid_tool_calls or []):  # type: ignore
                    result.append(
                        ToolMessage(
                            content="工具调用格式错误: %s"
                            % (invalid.get("error") or "参数解析失败"),
                            tool_call_id=invalid.get("id") or f"{INVALID_ID_PREFIX}{base_index}_{k}",
                            name=invalid.get("name") or "unknown_tool",
                        )
                    )
            return {"messages": result}

        async def tools_node(
            state: AgentState, config: RunnableConfig, runtime: Runtime
        ) -> dict[str, list[BaseMessage]]:
            # 工具节点：按每次 run 的 context 工具集构建 ToolNode 并执行
            # （ToolNode 是普通 Runnable，ainvoke 返回 {"messages": [...]} 更新）。
            context = cast(AgentContext, runtime.context)
            tool_node = ToolNode(context.tools)
            return await tool_node.ainvoke(state, config)

        graph = StateGraph(AgentState, context_schema=AgentContext)
        # runtime 注入在 langgraph 类型定义之外（运行时已验证），类型检查忽略
        graph.add_node(AGENT_NODE, agent_node)  # type: ignore
        graph.add_node(TOOLS_NODE, tools_node)  # type: ignore
        graph.add_edge(START, AGENT_NODE)
        graph.add_conditional_edges(AGENT_NODE, should_continue)
        graph.add_edge(TOOLS_NODE, AGENT_NODE)
        return graph.compile(checkpointer=self._checkpointer)

    def _run_config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._config.recursion_limit,
        }

    def _initial_state(self, query: str, system: Optional[str]) -> dict[str, Any]:
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
        state = await self._graph.aget_state(config)
        current: list[BaseMessage] = (state.values or {}).get("messages", [])
        if not current:
            return
        repaired = repair_state_for_checkpoint(list(current))
        if repaired == current:
            return
        await self._graph.aupdate_state(
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

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """attempt（已执行次数）后判定是否重试：白名单内且未达上限。"""
        return attempt < self._config.retry_max_attempts and is_retryable(
            exc, self._config.retryable_exceptions
        )

    async def _wait_before_retry(self, exc: Exception, attempt: int) -> None:
        delay = compute_delay(
            attempt,
            self._config.retry_base_delay,
            self._config.retry_backoff_factor,
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
        state = await self._graph.aget_state(config)
        if state.next or state.values:
            await self._repair_checkpoint_state(thread_id)
            return None
        return initial

    async def invoke(
        self,
        query: str,
        *,
        thread_id: str,
        system: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> ConversationResult:
        """同步语义的一次调用：返回最终文本、本轮消息与工具调用汇总。

        messages 与 tool_calls 都只含本轮（最后一条 HumanMessage 起）产生的内容，
        不含该 thread 的历史轮次。context 可覆盖默认的 LLM/工具集（每次 run 注入）。
        """
        ctx = context or self._context
        await self._repair_checkpoint_state(thread_id)
        initial = cast(AgentState, self._initial_state(query, system))
        config = self._run_config(thread_id)
        attempt = 0
        while True:
            attempt += 1
            graph_input = (
                initial if attempt == 1 else await self._resume_input(thread_id, config, initial)
            )
            try:
                state = await self._graph.ainvoke(graph_input, config, context=ctx)
                break
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._wait_before_retry(exc, attempt)
        round_messages = messages_after_last_human(state["messages"])
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
        )

    async def stream(
        self,
        query: str,
        *,
        thread_id: str,
        system: Optional[str] = None,
        context: Optional[AgentContext] = None,
    ) -> AsyncIterator[AgentEvent]:
        """流式执行：依次产出 llm_token / tool_call / tool_result，最后 done 或 error。

        基于 graph.astream(stream_mode=["messages", "updates"])：
        - messages 通道给 token 级文本（只取 agent 节点的 chunk）
        - updates 通道给完整 AIMessage（tool_call 决策）与 ToolMessage（tool_result）
        context 可覆盖默认的 LLM/工具集（每次 run 注入，重试续跑沿用同一份）。
        """
        ctx = context or self._context
        final_text = ""
        try:
            await self._repair_checkpoint_state(thread_id)
            initial = cast(AgentState, self._initial_state(query, system))
            config = self._run_config(thread_id)
            attempt = 0
            while True:
                attempt += 1
                graph_input = (
                    initial if attempt == 1 else await self._resume_input(thread_id, config, initial)
                )
                try:
                    async for mode, payload in self._graph.astream(
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
            state = await self._graph.aget_state(config)
            # 只汇总本轮（最后一条 HumanMessage 起）的 tool_calls，不含历史轮次
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
        except Exception as exc:  # noqa: BLE001 统一转 error 事件，不外泄 traceback
            yield AgentEvent(EVENT_ERROR, {"message": f"{type(exc).__name__}: {exc}"})
