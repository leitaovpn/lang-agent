"""core 层：手搓 StateGraph 的完整 ReAct agent loop。

拓扑：
    START → agent（LLM + bind_tools，流式合并）→ tools_condition
              ├─ 无 tool_calls → END
              └─ 有 tool_calls → tools（ToolNode 执行真实工具）→ agent（回环）

用 checkpointer 按 thread_id 记忆多轮对话；对外统一暴露 invoke / stream，
屏蔽底层 agent（图结构、工具、LLM）差异。

注意：本模块不能使用 `from __future__ import annotations`——langgraph 会用本模块的
globals 求值 AgentState 的 `Annotated[...]` 注解，字符串化会导致 NameError。
"""
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
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

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
from lang_agent.core.repair import INVALID_ID_PREFIX, repair_messages_for_llm
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


@dataclass
class AgentLoopConfig:
    checkpointer_kind: str = "memory"  # "memory" | "sqlite"
    db_path: str = "~/.lang-agent/checkpoints.sqlite"
    recursion_limit: int = 25


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
        self._graph = self._build_graph()

    def _build_tools(self, tools:  Optional[list[BaseTool]] = None) -> list[BaseTool]:
        """按配置构建工具列表（可被子类覆盖）。"""
        default_tools = instantiate_tools()
        return default_tools + (tools or [])

    def _build_graph(self):
        model = self._llm.bind_tools(self._tools)

        async def agent_node(
            state: AgentState, config: RunnableConfig
        ) -> dict[str, list[BaseMessage]]:
            # agent 节点：流式调用 LLM 并合并 chunk，保证 token 级事件可被捕获。
            # 必须把 config 传给 astream：否则 bind_tools 的 RunnableBinding 内层
            # 模型收不到回调，token 级流式事件会丢失（langgraph 0.6 行为）。
            # 发给 LLM 的历史先过修复层：保证每条 tool_call 都有对应 ToolMessage、
            # tool_call_id 不重复（LLM 接口的硬性要求）。
            messages: list[BaseMessage] = repair_messages_for_llm(list(state["messages"]))
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
                            tool_call_id=f"{INVALID_ID_PREFIX}{base_index}_{k}",
                            name=invalid.get("name") or "unknown_tool",
                        )
                    )
            return {"messages": result}

        graph = StateGraph(AgentState)
        graph.add_node(AGENT_NODE, agent_node)
        graph.add_node(TOOLS_NODE, ToolNode(self._tools))
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

    async def invoke(
        self, query: str, *, thread_id: str, system: Optional[str] = None
    ) -> ConversationResult:
        """同步语义的一次调用：返回最终文本、本轮消息与工具调用汇总。

        messages 与 tool_calls 都只含本轮（最后一条 HumanMessage 起）产生的内容，
        不含该 thread 的历史轮次。
        """
        state = await self._graph.ainvoke(
            cast(AgentState, self._initial_state(query, system)), self._run_config(thread_id)
        )
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
        self, query: str, *, thread_id: str, system: Optional[str] = None
    ) -> AsyncIterator[AgentEvent]:
        """流式执行：依次产出 llm_token / tool_call / tool_result，最后 done 或 error。

        基于 graph.astream(stream_mode=["messages", "updates"])：
        - messages 通道给 token 级文本（只取 agent 节点的 chunk）
        - updates 通道给完整 AIMessage（tool_call 决策）与 ToolMessage（tool_result）
        """
        final_text = ""
        try:
            async for mode, payload in self._graph.astream(
                cast(AgentState, self._initial_state(query, system)),
                self._run_config(thread_id),
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
                                ):
                                    final_text = message.content
                        for event in classify_node_update(node, delta):
                            yield event
            state = await self._graph.aget_state(self._run_config(thread_id))
            # 只汇总本轮（最后一条 HumanMessage 起）的 tool_calls，不含历史轮次
            tool_calls = collect_tool_calls(
                messages_after_last_human(state.values.get("messages", []))
            )
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
