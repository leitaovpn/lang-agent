"""core 层：ReAct agent loop 的纯 graph 薄封装。

拓扑：
    START → agent（LLM + bind_tools，流式合并）
              ├─ 无 tool_calls → END
              ├─ 只有 invalid_tool_calls（已附错误反馈）→ agent（重试）
              └─ 有 tool_calls → approval（可中断）→ tools（仅执行获批工具）→ agent

AgentLoop 只负责构建与编译 graph；invoke/stream 与 `graph.ainvoke/astream`
完全同形透传（输入/输出均不塑形）。llm/tools 每次 run 经 `context=` 注入
（AgentContext），本模块不持有任何 LLM/工具——invoke/stream 的 context 为必传。

checkpoint 修复、context 压缩、重试续跑、结果塑形与事件分类等入口逻辑
在 agent 层（lang_agent/agent/orchestration/session.py），经 `AgentLoop.graph` 访问
aget_state/aupdate_state 实现。

注意：本模块不能使用 `from __future__ import annotations`——langgraph 会用本模块的
globals 求值 AgentState 的 `Annotated[...]` 注解，字符串化会导致 NameError
（1.2.11 仍是该求值机制）。
"""
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolCallWrapper
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict

from .approval import (
    ToolApprovalDecision,
    ToolApprovalHook,
    ToolApprovalItem,
    approve_all,
    build_approval_request,
    validate_approval_decision,
)
from .compress import truncate_tool_outputs
from .repair import INVALID_ID_PREFIX

AGENT_NODE = "agent"
TOOLS_NODE = "tools"
APPROVAL_NODE = "approval"


class AgentState(TypedDict):
    """agent 循环状态：messages 用 add_messages reducer 累加；system 为每线程静态元数据。"""

    messages: Annotated[list[AnyMessage], add_messages]
    system: str
    raw_input: str
    summary: str  # 早期对话摘要（context 压缩写回；发送时作为 SystemMessage 前缀）
    tool_approval: ToolApprovalDecision


def should_continue(state: AgentState) -> str:
    """循环条件：以最近一条 AIMessage 决定走向（工具结果/错误反馈由固定边处理）。

    - 有合法 tool_calls → approval（门禁通过后才执行工具）
    - 只有 invalid_tool_calls（agent 节点已附错误反馈 ToolMessage）→ agent 重试
    - 纯文本回答 → END
    """
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            if message.tool_calls:
                return APPROVAL_NODE
            if message.invalid_tool_calls:
                return AGENT_NODE
            return END
    return END


class AgentContext(BaseModel):
    """每次 run 的静态上下文：LLM 与工具集（每次 invoke/stream 经 context= 注入）。

    langgraph 1.x 的 context 特性：不写入 checkpoint、run 内只读、共享给所有节点。
    节点通过注入的 runtime 对象访问（runtime.context）——1.2.11 仍不支持以
    `context` 参数名直接注入节点函数。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: BaseChatModel
    tools: list[BaseTool]
    summarizer_llm: BaseChatModel | None = None  # context 压缩的摘要模型，缺省复用 llm
    # 工具执行拦截器（可选）：由 agent 层构建 context 时决定是否注入，
    # 存在时默认 tools_node 构建 ToolNode 会传入 wrap_tool_call（重试/缓存/鉴权挂点）
    wrap_tool_call: ToolCallWrapper | None = None


@dataclass(slots=True)
class AgentLoopConfig:
    """agent 循环配置（被动数据，逐字段标注消费层）。"""

    # core 消费：checkpointer 构建（build_checkpointer）
    checkpointer_kind: str = "memory"  # "memory" | "sqlite"
    db_path: str = "~/.lang-agent/checkpoints.sqlite"
    # core 消费：默认 agent 节点构造时闭包捕获（构造后改配置不生效）
    compress_tool_output_max_chars: int = 2000  # 发送视图工具输出截断上限
    # agent 层消费：_run_config
    recursion_limit: int = 25
    # agent 层消费：graph 调用异常重试（瞬时异常白名单 + 指数退避）
    retry_max_attempts: int = 3  # 总尝试次数（含首次）
    retry_base_delay: float = 0.5  # 首次退避秒数
    retry_backoff_factor: float = 2.0  # 指数退避因子
    retryable_exceptions: tuple[type[BaseException], ...] | None = None  # None → 默认白名单（见 core/retry.py）
    # agent 层消费：context 压缩（摘要 + 保留窗口 + 工具输出截断，见 core/compress.py）
    compress_enabled: bool = True
    compress_token_threshold: int = 8000  # 估算 token 超此阈值触发压缩
    compress_keep_last: int = 8  # 保留最近 N 条完整消息
    # core 消费：工具执行门禁。None 保持自动放行；应用层默认注入人工审批钩子。
    tool_approval_hook: ToolApprovalHook | None = None


ReActNode = Callable[
    [AgentState, RunnableConfig, Runtime[AgentContext]],
    Awaitable[dict[str, Any]],
]
"""ReAct 节点签名：state/config 由 langgraph 注入，runtime.context 取 AgentContext。"""


def build_default_agent_node(compress_tool_output_max_chars: int = 2000) -> ReActNode:
    """默认 agent 节点：流式调用 context 的 LLM 并合并 chunk，保证 token 级事件可被捕获。

    - 必须把 config 传给 astream：否则 bind_tools 的 RunnableBinding 内层
      模型收不到回调，token 级流式事件会丢失（1.x 仍是该机制）。
    - LLM 与工具集从 runtime.context 取（每次 run 注入，见 AgentContext）。
    - chunk 合并后转普通 AIMessage 并用严格 json 从 tool_call_chunks 重建
      合法/非法判定（langchain-core 1.x 的 chunk 合并/序列化会把
      invalid_tool_calls 误判为合法调用，详见 AGENTS.md 已知坑）。
    """

    async def agent_node(
        state: AgentState, config: RunnableConfig, runtime: Runtime[AgentContext]
    ) -> dict[str, list[BaseMessage]]:
        context = cast(AgentContext, runtime.context)
        model = context.llm.bind_tools(context.tools)
        # 发送视图：system 提示 + 摘要前缀 + 截断后的历史（截断不写回 checkpoint）
        messages: list[BaseMessage] = list(state["messages"])
        prefix: list[BaseMessage] = []
        if state.get("system"):
            prefix.append(SystemMessage(content=state["system"]))
        if state.get("summary"):
            prefix.append(SystemMessage(content="早期对话摘要:\n" + state["summary"]))
        messages = truncate_tool_outputs(prefix + messages, compress_tool_output_max_chars)
        chunks: list[BaseMessageChunk] = []
        async for chunk in model.astream(messages, config=config):
            chunks.append(cast(BaseMessageChunk, chunk))
        if not chunks:
            return {"messages": []}
        merged = cast(AIMessageChunk, chunks[0])
        for other in chunks[1:]:
            merged = merged + cast(AIMessageChunk, other)
        # langchain-core 1.x：chunk 合并（add_ai_message_chunks）会用合并后的
        # tool_call_chunks 重新构造 chunk，触发 init_tool_calls 校验器——它对
        # 残缺 args 宽容解析为 {}，把 invalid_tool_calls 误判为合法 tool_calls
        # （0.3.x 无此行为）。tool_call_chunks 保留原始 args 字符串，这里用严格
        # json 解析重建合法/非法判定（合法调用的最终 args 必为完整 JSON）。
        tool_calls: list[dict[str, Any]] = []
        invalid_tool_calls: list[dict[str, Any]] = []
        for raw in merged.tool_call_chunks:
            name = raw.get("name") or ""
            args_raw = raw.get("args")
            if isinstance(args_raw, dict):
                args: Any = args_raw
            elif args_raw:
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = None
            else:
                args = {}
            if isinstance(args, dict):
                tool_calls.append(
                    {"name": name, "args": args, "id": raw.get("id"), "type": "tool_call"}
                )
            else:
                invalid_tool_calls.append(
                    {
                        "name": name,
                        "args": args_raw or "",
                        "id": raw.get("id"),
                        "type": "invalid_tool_call",
                        "error": None,
                    }
                )
        # 转成普通 AIMessage 返回：AIMessageChunk（无论合并与否）经 checkpointer
        # 序列化往返同样会触发 init_tool_calls 重建造成误判，普通 AIMessage
        # 无此校验器，invalid_tool_calls 可原样往返。
        # invalid 的 id 用确定性 invalid_<aimessage下标>_<条内序号>：agent_node
        # 合成反馈 ToolMessage 与 repair 层必须一致，且 feedback ToolMessage 的
        # tool_call_id 必须等于 _convert_message_to_dict 发给 API 的 tool_calls id
        # （invalid_tool_calls 会被序列化为 tool_calls 发送）——原始 id（call_9 之类）
        # 会造成「tool_calls 后缺匹配 ToolMessage」，deepseek 校验报 400。
        base_index = len(state["messages"])
        for k, invalid in enumerate(invalid_tool_calls or []):
            invalid = dict(invalid)
            invalid["id"] = f"{INVALID_ID_PREFIX}{base_index}_{k}"
            invalid_tool_calls[k] = invalid
        message = AIMessage(
            content=merged.content,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            additional_kwargs=merged.additional_kwargs,
            response_metadata=merged.response_metadata,
            usage_metadata=merged.usage_metadata,
            id=merged.id,
            name=merged.name,
        )
        result: list[BaseMessage] = [message]
        if not message.tool_calls:
            # 无合法调用：为解析失败的调用附错误反馈 ToolMessage，驱动循环重试。
            # 注意 invalid_tool_calls 在 AImessage 里的 id 已被重写为确定性 id
            # （与 repair 层统一），这里直接用 message.invalid_tool_calls 的 id。
            for raw_invalid in message.invalid_tool_calls or []:
                invalid = dict(raw_invalid)
                result.append(
                    ToolMessage(
                        content=f"工具调用格式错误: {invalid.get('error') or '参数解析失败'}",
                        tool_call_id=invalid["id"],
                        name=invalid.get("name") or "unknown_tool",
                    )
                )
        return {"messages": result}

    return agent_node


def build_default_tools_node() -> ReActNode:
    """默认工具节点：按每次 run 的 context 工具集构建 ToolNode 并执行。

    必须显式 handle_tool_errors=True：1.x 默认只把 ToolInvocationError 转
    错误 ToolMessage，其余工具异常直接上抛；显式 True 恢复 0.6.x 语义
    （任何工具异常 → 错误 ToolMessage 投喂 LLM，循环继续）。
    context.wrap_tool_call 存在时传入 ToolNode（工具拦截：重试/缓存/鉴权）。
    """

    async def tools_node(
        state: AgentState, config: RunnableConfig, runtime: Runtime[AgentContext]
    ) -> dict[str, Any]:
        context = cast(AgentContext, runtime.context)
        approval: ToolApprovalDecision | None = state.get("tool_approval")
        decisions: dict[str, ToolApprovalItem] = {}
        if approval:
            decisions = {
                item["tool_call_id"]: item for item in approval["decisions"]
            }
        latest_index = None
        for index in range(len(state["messages"]) - 1, -1, -1):
            candidate = state["messages"][index]
            if isinstance(candidate, AIMessage) and candidate.tool_calls:
                latest_index = index
                break
        if latest_index is None:
            return {"messages": [], "tool_approval": {}}
        latest = cast(AIMessage, state["messages"][latest_index])
        approved = []
        rejected = []
        for call in latest.tool_calls:
            item = decisions.get(str(call.get("id")))
            if item is None:
                continue
            if item["action"] == "approve":
                approved.append(call)
            else:
                rejected.append(call)
        messages: list[BaseMessage] = []
        if approved:
            approved_ai = latest.model_copy(update={"tool_calls": approved})
            tool_state = dict(state)
            tool_state["messages"] = [*state["messages"][:latest_index], approved_ai]
            tool_node = ToolNode(
                context.tools,
                handle_tool_errors=True,
                wrap_tool_call=context.wrap_tool_call,
            )
            result = await tool_node.ainvoke(tool_state, config)
            messages.extend(result.get("messages", []))
        for call in rejected:
            item = decisions[str(call.get("id"))]
            reason = item.get("reason") or "用户拒绝执行"
            messages.append(
                ToolMessage(
                    content=f"工具调用未执行：{reason}",
                    tool_call_id=str(call.get("id") or ""),
                    name=str(call.get("name") or "unknown_tool"),
                    status="error",
                )
            )
        return {"messages": messages, "tool_approval": {}}

    return tools_node


def build_default_approval_node(hook: ToolApprovalHook | None = None) -> ReActNode:
    """构建工具审批节点；门禁决定完整有效后才允许进入工具节点。"""
    gate = hook or approve_all

    async def approval_node(
        state: AgentState, config: RunnableConfig, runtime: Runtime[AgentContext]
    ) -> dict[str, Any]:
        del config, runtime
        latest = next(
            (
                message
                for message in reversed(state["messages"])
                if isinstance(message, AIMessage) and message.tool_calls
            ),
            None,
        )
        if latest is None:
            return {"tool_approval": {}}
        request = build_approval_request([dict(call) for call in latest.tool_calls])
        decision = gate(request)
        if inspect.isawaitable(decision):
            decision = await decision
        return {"tool_approval": validate_approval_decision(request, decision)}

    return approval_node


async def build_checkpointer(config: AgentLoopConfig):
    """按配置构建 checkpointer（异步工厂）。

    sqlite 用 AsyncSqliteSaver，其构造绑定当前事件循环，必须在异步上下文中创建；
    memory 用 InMemorySaver，可在任意位置创建。
    """
    if config.checkpointer_kind == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        if config.db_path != ":memory:":
            # expanduser：os.path.abspath 不展开 ~，会在工作目录下建出字面量 ~ 目录
            db_path = Path(config.db_path).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(db_path))
        else:
            conn = await aiosqlite.connect(config.db_path)
        return AsyncSqliteSaver(conn)
    if config.checkpointer_kind == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    raise ValueError(f"未知 checkpointer_kind: {config.checkpointer_kind!r}")


class AgentLoop:
    """ReAct agent 循环的纯 graph 薄封装。

    - 构造时只编译 graph（checkpointer、config、可选注入的 agent_node/tools_node）；
      不持有任何 LLM/工具。
    - invoke/stream 与 `graph.ainvoke/astream` 完全同形透传：输入/输出均不塑形，
      stream 产出原始 (mode, payload) 元组（stream_mode 为列表时）。
    - llm/tools 每次 run 经 `context=`（AgentContext）必传注入，由调用方
      （agent 层）构造；缺省节点工厂见 build_default_agent_node / build_default_tools_node。
    """

    def __init__(
        self,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
        config: AgentLoopConfig | None = None,
        agent_node: ReActNode | None = None,
        tools_node: ReActNode | None = None,
        approval_node: ReActNode | None = None,
    ) -> None:
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
        # 默认节点在构造时捕获 config 的截断参数（构造后改配置不生效）
        self._graph = self._build_graph(
            agent_node
            or build_default_agent_node(self._config.compress_tool_output_max_chars),
            approval_node or build_default_approval_node(self._config.tool_approval_hook),
            tools_node or build_default_tools_node(),
        )

    def _build_graph(
        self, agent_node: ReActNode, approval_node: ReActNode, tools_node: ReActNode
    ) -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
        graph = StateGraph(AgentState, context_schema=AgentContext)
        # runtime 注入在 langgraph 类型定义之外（运行时已验证），类型检查忽略
        graph.add_node(AGENT_NODE, agent_node)  # type: ignore
        graph.add_node(APPROVAL_NODE, approval_node)  # type: ignore
        graph.add_node(TOOLS_NODE, tools_node)  # type: ignore
        graph.add_edge(START, AGENT_NODE)
        graph.add_conditional_edges(AGENT_NODE, should_continue)
        graph.add_edge(APPROVAL_NODE, TOOLS_NODE)
        graph.add_edge(TOOLS_NODE, AGENT_NODE)
        # langgraph 不同补丁版本对 compile() 的 input/output StateT 推断不同；
        # 本图未声明独立 input/output schema，二者在运行时均为 AgentState。
        return cast(
            CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState],
            graph.compile(checkpointer=self._checkpointer),
        )

    @property
    def graph(self) -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
        """暴露编译图：agent 层 checkpoint 修复/压缩/续跑需要 aget_state/aupdate_state。"""
        return self._graph

    async def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: AgentContext | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """graph.ainvoke 的完全同形透传（input/config/context/其余 kwargs）。

        input 为 graph 的初始 state（部分 AgentState dict 亦可，langgraph 运行时补全）。
        """
        _require_context(context)
        return await self._graph.ainvoke(input, config, context=context, **kwargs)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: AgentContext | None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """graph.astream 的完全同形透传（普通函数，直接返回其异步迭代器）。

        产出原始 (mode, payload) 元组（stream_mode 为列表时），不做事件分类。
        注意：不是 async 函数，不要 `await loop.stream(...)`——直接 async for 消费。
        """
        _require_context(context)
        return self._graph.astream(input, config, context=context, **kwargs)


def _require_context(context: AgentContext | None) -> None:
    """context 必传的运行时防线：langgraph 对 None 静默放行（_coerce_context），
    节点里才会以晦涩的 AttributeError 暴露——这里提前给出明确错误。"""
    if context is None:
        raise ValueError(
            "AgentLoop 不持有默认 LLM/工具：invoke/stream 必须显式传入 context"
            "（AgentContext(llm=..., tools=...)），由调用方（agent 层）构造注入"
        )
