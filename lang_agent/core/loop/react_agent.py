"""core 层：ReAct agent loop 的纯 graph 薄封装。

拓扑：
    START → before_agent → before_model → agent → after_model
              ├─ 完成 → after_agent → END
              ├─ invalid/重试 → before_model
              └─ 工具调用 → before_tool → tools → after_tool → before_model

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

import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict, cast
from uuid import uuid4

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

from lang_agent.plugin import (
    ModelRequest,
    ModelResponse,
    PluginError,
    PluginSpecSnapshot,
)
from lang_agent.plugin.graph import merge_namespaces
from lang_agent.plugin.runtime import CompiledPluginBundle, PluginRuntime
from lang_agent.plugin.types import NODE_HOOKS
from lang_agent.plugin.wrappers import call_sync_wrapper, compose_wrappers

from .compress import truncate_tool_outputs
from .repair import INVALID_ID_PREFIX

AGENT_NODE = "agent"
TOOLS_NODE = "tools"


class AgentState(TypedDict):
    """agent 循环状态：messages 用 add_messages reducer 累加；system 为每线程静态元数据。"""

    messages: Annotated[list[AnyMessage], add_messages]
    system: str
    raw_input: str
    agent_id: NotRequired[str]
    run_id: NotRequired[str]
    plugin_revision: NotRequired[str]
    plugin_state: Annotated[dict[str, dict[str, Any]], merge_namespaces]
    model_route_override: NotRequired[str | None]
    summary: str  # 早期对话摘要（context 压缩写回；发送时作为 SystemMessage 前缀）


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
    """每次 run 的静态上下文：LLM 与工具集（每次 invoke/stream 经 context= 注入）。

    langgraph 1.x 的 context 特性：不写入 checkpoint、run 内只读、共享给所有节点。
    节点通过注入的 runtime 对象访问（runtime.context）——1.2.11 仍不支持以
    `context` 参数名直接注入节点函数。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    agent_id: str | None = None
    plugin_bundle: Any = None
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
    recursion_limit: int = 80
    # agent 层消费：graph 调用异常重试（瞬时异常白名单 + 指数退避）
    retry_max_attempts: int = 3  # 总尝试次数（含首次）
    retry_base_delay: float = 0.5  # 首次退避秒数
    retry_backoff_factor: float = 2.0  # 指数退避因子
    retryable_exceptions: tuple[type[BaseException], ...] | None = (
        None  # None → 默认白名单（见 core/retry.py）
    )
    # agent 层消费：context 压缩（摘要 + 保留窗口 + 工具输出截断，见 core/compress.py）
    compress_enabled: bool = True
    compress_token_threshold: int = 8000  # 估算 token 超此阈值触发压缩
    compress_keep_last: int = 8  # 保留最近 N 条完整消息


ReActNode = Callable[
    [AgentState, RunnableConfig, Runtime[AgentContext]],
    Awaitable[dict[str, list[BaseMessage]]],
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
        # 发送视图：system 提示 + 摘要前缀 + 截断后的历史（截断不写回 checkpoint）
        messages: list[BaseMessage] = copy.deepcopy(list(state["messages"]))
        prefix: list[BaseMessage] = []
        if state.get("system"):
            prefix.append(SystemMessage(content=state["system"]))
        if state.get("summary"):
            prefix.append(SystemMessage(content="早期对话摘要:\n" + state["summary"]))
        messages = truncate_tool_outputs(
            prefix + messages, compress_tool_output_max_chars
        )

        async def terminal(request: ModelRequest) -> ModelResponse:
            if any(tool not in context.tools for tool in request.tools):
                raise PluginError("模型请求包含 context.tools 之外的工具")
            model = request.llm.bind_tools(request.tools)
            chunks: list[BaseMessageChunk] = []
            async for chunk in model.astream(
                request.messages,
                config={
                    **request.config,
                    "metadata": {
                        **request.config.get("metadata", {}),
                        "plugin_model_role": "primary",
                    },
                },
            ):
                chunks.append(cast(BaseMessageChunk, chunk))
            if not chunks:
                return ModelResponse(None)
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
                        {
                            "name": name,
                            "args": args,
                            "id": raw.get("id"),
                            "type": "tool_call",
                        }
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
            return ModelResponse(message)

        entries = (
            context.plugin_bundle.snapshot.registrations
            if context.plugin_bundle
            else ()
        )
        request = ModelRequest(
            context.llm, list(context.tools), messages, config, runtime
        )
        response = await compose_wrappers(entries, "wrap_model_hook", terminal)(request)
        if not isinstance(response, ModelResponse):
            raise PluginError("模型 wrapper 必须返回 ModelResponse")
        if response.message is None:
            # 空短路也是本次模型决策，避免沿用上一条 tool_calls 重复执行工具。
            return {"messages": [AIMessage(content="")]}
        if not isinstance(response.message, AIMessage) or isinstance(
            response.message, AIMessageChunk
        ):
            raise PluginError("模型响应必须是普通 AIMessage")
        message = AIMessage.model_validate(response.message.model_dump())
        for k, invalid_call in enumerate(message.invalid_tool_calls):
            invalid_call["id"] = f"{INVALID_ID_PREFIX}{len(state['messages'])}_{k}"
        result: list[BaseMessage] = [message]
        if message.invalid_tool_calls:
            # 解析失败：为解析失败的调用附错误反馈 ToolMessage，驱动循环重试。
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
    插件与旧 context.wrap_tool_call 合成为异步 wrapper，兼容仅异步工具。
    """

    async def tools_node(
        state: AgentState, config: RunnableConfig, runtime: Runtime[AgentContext]
    ) -> dict[str, list[BaseMessage]]:
        context = cast(AgentContext, runtime.context)
        entries = (
            context.plugin_bundle.snapshot.registrations
            if context.plugin_bundle
            else ()
        )
        plugin_names = tuple(r.name for r in entries)
        contract_errors: list[PluginError] = []

        async def wrapper(request, execute):
            from lang_agent.plugin.approval import enforce_approval
            from lang_agent.plugin.tool_validation import validate_tool_result

            original_id = request.tool_call["id"]
            request = request.override(
                tool_call=copy.deepcopy(request.tool_call), state=copy.deepcopy(state)
            )
            denial = enforce_approval(state, request, plugin_names=plugin_names)
            if denial is not None:
                return denial

            async def terminal(req):
                enforce_approval(state, req, plugin_names=plugin_names, executing=True)
                if req.tool_call.get("id") != original_id:
                    raise PluginError("wrapper 不能修改 tool_call_id")
                named = next(
                    (t for t in context.tools if t.name == req.tool_call["name"]), None
                )
                if req.tool_call["name"] != request.tool_call["name"] and named is None:
                    raise PluginError("wrapper 指定了未注册的工具")
                req = req.override(tool=named)
                return await execute(req)

            async def legacy(req):
                if context.wrap_tool_call:
                    return await call_sync_wrapper(
                        context.wrap_tool_call, req, terminal
                    )
                return await terminal(req)

            result = await compose_wrappers(entries, "wrap_tool_hook", legacy)(request)
            validate_tool_result(result, original_id)
            return result

        async def checked_wrapper(request, execute):
            try:
                return await wrapper(request, execute)
            except PluginError as exc:
                contract_errors.append(exc)
                raise

        tool_node = ToolNode(
            context.tools,
            handle_tool_errors=True,
            awrap_tool_call=checked_wrapper,
        )
        result = await tool_node.ainvoke(state, config)
        if contract_errors:
            raise contract_errors[0]
        return result

    return tools_node


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

    - 构造时只编译 graph（checkpointer、config）；节点固定用默认工厂
      build_default_agent_node / build_default_tools_node，不支持注入自定义节点
      （旁路 wrap hook 的执行保证，见 docs/superpowers/specs/plugin.md 第 15 节）；
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
    ) -> None:
        self._agent_id = uuid4().hex
        self._plugins = PluginRuntime(self._agent_id)
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
        # 固定使用内置默认节点：自定义节点会旁路 wrap hook 的执行保证，
        # 不支持注入（见 docs/superpowers/specs/plugin.md 第 15 节）。
        # 默认节点在构造时捕获 config 的截断参数（构造后改配置不生效）。
        self._graph = self._build_graph(
            build_default_agent_node(self._config.compress_tool_output_max_chars),
            build_default_tools_node(),
        )

    def _build_graph(
        self, agent_node: ReActNode, tools_node: ReActNode
    ) -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
        # 显式钉住四个泛型参数：StateGraph 为 StateT/ContextT/InputT/OutputT
        # 四参泛型，仅靠构造器只绑定前两个，compile() 返回类型里 InputT/
        # OutputT 悬空，旧版 mypy 无法用声明返回类型反推（report-return-value）。
        graph: StateGraph[AgentState, AgentContext, AgentState, AgentState] = (
            StateGraph(AgentState, context_schema=AgentContext)
        )

        # runtime 位置注入在 langgraph 类型定义之外（运行时已验证）：
        # 1.2.x 的 StateNode 只有 (state, *, runtime) 形式，不支持
        # (state, config, runtime) 组合（Runtime 不含 config），
        # 故 add_node 处用 cast 显式豁免类型检查。
        def guarded(node):
            async def execute(
                state: AgentState,
                config: RunnableConfig,
                runtime: Runtime[AgentContext],
            ):
                self._plugins.validate(state, runtime.context)
                return await node(state, config, runtime)

            return execute

        graph.add_node(AGENT_NODE, cast(Any, guarded(agent_node)))
        graph.add_node(TOOLS_NODE, cast(Any, guarded(tools_node)))
        for hook in NODE_HOOKS:

            def dispatcher(name):
                async def execute(
                    state: AgentState,
                    config: RunnableConfig,
                    runtime: Runtime[AgentContext],
                ):
                    return await self._plugins.dispatch(
                        name, state, config, runtime.context
                    )

                return execute

            graph.add_node(hook, cast(Any, dispatcher(hook)))
        graph.add_edge(START, "before_agent")
        graph.add_edge("before_agent", "before_model")
        graph.add_edge("before_model", AGENT_NODE)
        graph.add_edge(AGENT_NODE, "after_model")

        def route(state: AgentState):
            override = state.get("model_route_override")
            if override:
                return "before_model" if override == "retry" else "after_agent"
            return {
                AGENT_NODE: "before_model",
                TOOLS_NODE: "before_tool",
                END: "after_agent",
            }[should_continue(state)]

        graph.add_conditional_edges(
            "after_model", route, ["before_model", "before_tool", "after_agent"]
        )
        graph.add_edge("before_tool", TOOLS_NODE)
        graph.add_edge(TOOLS_NODE, "after_tool")
        graph.add_edge("after_tool", "before_model")
        graph.add_edge("after_agent", END)
        return graph.compile(checkpointer=self._checkpointer)

    @property
    def agent_id(self) -> str:
        """新建时生成、跨轮次不变的 agent 身份。"""
        return self._agent_id

    @classmethod
    def restore(cls, *, agent_id: str, **kwargs: Any) -> "AgentLoop":
        """从受信任的持久定义恢复既有身份；插件历史快照需重新装载。"""
        if not agent_id:
            raise ValueError("agent_id 不能为空")
        loop = cls(**kwargs)
        loop._agent_id = agent_id
        loop._plugins = PluginRuntime(agent_id)
        return loop

    @property
    def plugin_revision(self) -> str:
        return self._plugins.get().revision

    def update_plugin_hooks(
        self, snapshot: PluginSpecSnapshot, *, expected_revision: str | None = None
    ):
        return self._plugins.update(snapshot, expected_revision)

    def describe_plugin_hooks(self, revision: str | None = None) -> dict[str, Any]:
        bundle = self._plugins.get(revision)
        return {
            "agent_id": self.agent_id,
            "revision": bundle.revision,
            "hooks": {
                hook: [
                    {
                        "plugin": r.name,
                        "version": r.version,
                        "sync": bool(binding.sync),
                        "async": bool(binding.async_),
                    }
                    for r in bundle.snapshot.registrations
                    if (binding := r.hook(hook)) is not None
                ]
                for hook in (*NODE_HOOKS, "wrap_model_hook", "wrap_tool_hook")
            },
        }

    def validate_plugin_answers(
        self, revision: str, interruptions: list[Any], answers: dict[str, Any]
    ) -> None:
        """提交 Command 前校验答案，防止无效答案污染持久恢复记录。"""
        self._plugins.validate_answers(revision, interruptions, answers)

    def bind_plugin_context(
        self, context: AgentContext, *, revision: str | None = None
    ) -> AgentContext:
        _require_context(context)
        if context.agent_id not in (None, self.agent_id):
            raise PluginError("context 属于另一个 agent")
        bundle: CompiledPluginBundle = self._plugins.get(revision)
        return context.model_copy(
            update={"agent_id": self.agent_id, "plugin_bundle": bundle}
        )

    @property
    def graph(
        self,
    ) -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
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
