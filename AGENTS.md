# AGENTS.md

本文件为在此仓库工作的 AI 编码代理提供指导。

## 项目概览

lang-agent：基于 langgraph 的 lang agent，分三层：

- **ai**（LLM 接入）：`get_llm(model, provider, protocol)` 注册表工厂，映射为 langchain chat model 调用对象
- **core**（graph 薄封装）：`loop/` 子包（手搓 StateGraph 的完整 ReAct 循环 + repair/retry/compress/events 支撑；`AgentLoop` 构造只编译 graph（可注入节点），invoke/stream 与 graph 完全同形透传，不持有任何 LLM/工具）+ `tool/` 子包（工具实现与注册表）
- **agent**（会话入口 + 对外接口）：`orchestration/` 子包（`ChatSession`：repair/compress/retry/结果塑形/事件分类）+ `server/` 子包（FastAPI `/chat`、`/chat/stream` SSE）+ `cli/` 子包（纯 HTTP 客户端）

依赖方向严格单向 `agent → core → ai`，禁止反向 import。README.md 面向使用者（用法/API/配置），本文件面向在此仓库开发的代理。

## 开发命令

环境用仓库内 venv（`.venv`，Python 3.13.3），**不要用系统 python3**：

```bash
.venv/bin/pip install -r requirements-dev.txt   # 安装依赖（锁精确版本）
.venv/bin/python -m pytest                      # 全部测试
.venv/bin/python -m pytest tests/test_core/test_loop/test_react_agent.py::test_memory_second_round_sees_history   # 单个测试
.venv/bin/python -m lang_agent.agent.cli serve --port 8000          # 起服务
.venv/bin/python -m lang_agent.agent.cli chat --msg "..." --stream  # CLI 冒烟（一次性）
.venv/bin/python -m lang_agent.agent.cli chat                       # 交互模式（默认流式，自动拉起服务）
```

lint 与类型检查工具已锁进 dev 依赖：

```bash
.venv/bin/ruff check lang_agent/ tests/                                   # lint
.venv/bin/mypy lang_agent tests --explicit-package-bases                  # 类型检查
```

IDE 类型诊断与运行时同为 Python 3.13，可直接以诊断为准。

### 提交门禁（push / merge 前自动校验）

- **本地钩子**（`.githooks/`）：`pre-push` 与 `pre-merge-commit` 都会跑 ruff + mypy + pytest 全量校验，任一失败即中止。一次性启用：`git config core.hooksPath .githooks`（新克隆需重新执行）。
- **CI**（`.github/workflows/ci.yml`）：push 与 PR 双触发，跑同一套校验（ubuntu + Python 3.13）。
- **合并门禁**：PR 合并前要求 CI 通过，需在 GitHub 仓库 Settings → Branches 对 main 开启 branch protection，勾选 Require status checks（`checks` job）。本地 `git merge`（非 fast-forward）由 pre-merge-commit 钩子把关；ff 合并的分支已在 push 时被校验。

## 架构要点

- **State**：`AgentState` TypedDict，`messages: Annotated[list[AnyMessage], add_messages]`（`AnyMessage` 来自 langchain_core.messages，`add_messages` 来自 langgraph.graph.message），外加只读元数据字段 `system`、`raw_input`
- **AgentLoop 薄封装契约**（core/loop/react_agent.py）：构造只编译 graph——入参 `checkpointer` / `config` / `agent_node` / `tools_node`（节点缺省用模块级工厂 `build_default_agent_node(compress_tool_output_max_chars)` / `build_default_tools_node()`，签名 `ReActNode`）；`invoke(input, config, *, context, **kwargs)` / `stream(...)` 与 `graph.ainvoke/astream` **完全同形透传**（stream 是普通函数，产出原始 `(mode, payload)` 元组，不做事件分类）；**context 必传**（core 运行时断言，langgraph 对 None 静默放行）；`loop.graph` 暴露 `CompiledStateGraph`（agent 层 repair/compress 用 `aget_state/aupdate_state`）。默认节点在构造时捕获 config 截断参数（构造后改配置不生效）
- **LLM/tools 经 langgraph context 注入**：`AgentContext`（pydantic，`arbitrary_types_allowed=True`）承载 llm + tools（+ 可选 `summarizer_llm` + 可选 `wrap_tool_call` 工具拦截器），`StateGraph(AgentState, context_schema=AgentContext)` 声明（**不是** `compile()` 参数——0.6.11 实测报错，1.2.11 位置参数不变）；每次 run 经 `invoke/stream` 的 `context=` 注入（重试续跑沿用同一份），由 agent 层（`agent/orchestration/deps.py` 的 `get_deps`）按请求参数构造并缓存。**节点不能以 `context` 参数名接收**（1.2.11 的节点参数注入白名单仍无 `context`），正确姿势是接收注入的 `runtime` 对象，用 `runtime.context` 访问；工具节点按 `runtime.context.tools` 动态构建 `ToolNode(context.tools, handle_tool_errors=True, awrap_tool_call=组合拦截器).ainvoke(state, config)`（handle_tool_errors 必须显式 True，见已知坑 #7；旧 wrap_tool_call 为 None 时跳过兼容层）
- **context 压缩**（`core/loop/compress.py` 纯逻辑 + `ChatSession._compress_checkpoint_state`）：invoke/stream 入口在 repair 之后执行。token 估算（`estimate_tokens`：优先 `llm.get_num_tokens_from_messages`，**依赖 transformers 缺失时抛 ImportError，退回字符/4 兜底**——不要把异常静默吞成「跳过压缩」）超 `compress_token_threshold`（默认 8000）触发：保留最近 `compress_keep_last`（默认 8）条，**切点由 `find_round_start` 落在轮起点**（tool_call 段永不拆散，压缩后仍满足 repair 不变式），更早轮次交给 `summarizer_llm`（缺省复用对话 llm，注意会消耗其调用额度）总结，摘要**追加**到 `AgentState.summary` 并 RemoveMessage 删旧消息写回 checkpoint。发送视图：summary 作为 SystemMessage 前缀（在 system 提示之后）+ `truncate_tool_outputs` 截断超长工具输出（只缩 content 不写回）。参数可配：`compress_enabled` / `compress_token_threshold` / `compress_keep_last` / `compress_tool_output_max_chars`
- **拓扑**：固定六个 hook 调度节点包围 agent/tools；具体连线及版本语义见文末 Plugin 模式。
- **LLM 历史不变式**（`core/loop/repair.py`）：checkpoint 里存储的历史必须满足「AIMessage 的每条 tool_call 后面都有对应 ToolMessage，tool_call_id 在**两个 AIMessage 之间**不重合」。ChatSession.invoke/stream 开始前 `_repair_checkpoint_state` 用 `repair_state_for_checkpoint` 修复并写回——范围是最后一条**带 tool_calls** 的 AIMessage 起（坏段可能被本轮末尾的纯文本 AIMessage 推到前缀）；写回用 RemoveMessage 全删再加回，精确重建顺序（add_messages reducer 会把新消息追加到末尾，直接返回修复列表会打乱顺序）。前缀 tool_call_id 完全不参与去重；段内缺结果合成错误 ToolMessage、重复/孤儿丢弃、重复 id 改名、invalid_tool_calls 合成错误反馈；`should_continue` 看到只有 invalid_tool_calls 的 AIMessage 会带反馈回 agent 重试
- **流式**：`graph.astream(stream_mode=["messages", "updates"])` 双通道——messages 给 token 级内容（metadata 按 `langgraph_node` 过滤；chunk 的 `additional_kwargs.reasoning_content` 增量为 thinking_token、`content` 为 llm_token），updates 给完整 AIMessage（tool_call）与 ToolMessage（tool_result）；事件分类是 `core/loop/events.py` 的纯函数。thinking 捞回依赖 provider 层：langchain-openai 1.6 的 `_convert_delta_to_message_chunk` 丢弃 reasoning_content，`ai/chat/deepseek.py` 的 `DeepSeekChatModel` override `_convert_chunk_to_generation_chunk` 从原始 delta 取回附加到 chunk
- **多轮记忆**：`thread_id` + checkpointer。默认 sqlite（`AsyncSqliteSaver`，重启不丢），`memory` 兜底（`InMemorySaver`）。**sqlite 只能在异步上下文创建**：`await build_checkpointer(config)` 后注入 `AgentLoop(..., checkpointer=...)`，直接构造会抛 ValueError。checkpoint-sqlite 3.x 硬依赖 `sqlite-vec`（原生扩展，macOS 有预编译 wheel）；db_path 的 `~` 在 `build_checkpointer` 里经 pathlib `expanduser` 展开
- **graph 调用重试**（`core/loop/retry.py` 纯逻辑 + ChatSession 重试循环 + AgentLoopConfig 重试字段）：`ainvoke/astream` 异常后按**瞬时异常白名单**重试（默认 httpx 传输/超时 + `RetryableError` 标记；agent 层 orchestration/config.py 注入 openai 限流/连接/超时/5xx 类），指数退避（默认 0.5/1/2s，`retry_max_attempts=3`）。**重试用 `input=None` 从 checkpoint 续跑**——失败的超步重执行、输入消息不重复追加（已实测验证）；**续跑前（`_resume_input`）会先执行 `_repair_checkpoint_state` 修复 checkpoint 里的消息**（本轮中途产生的坏段在入口修复之后才写入，不修则续跑时 LLM 拿到未修复历史；已实测 aupdate_state 不会清掉 pending 任务）；首步即失败（无 checkpoint）时复用原输入。确定性错误不重试；耗尽后 invoke 抛异常、stream 出 error 事件。**注意分层**：openai SDK（3.x 默认 `max_retries=2`）对 5xx/429 自带内部重试，会先吸收掉短故障，我们的层只在 SDK 耗尽后接管
- **测试设施**：`tests/` 镜像包结构；LLM 一律用 `tests/conftest.py` 的 `FakeChatModel`（脚本队列驱动，`bind_tools` 返回真实 RunnableBinding），**测试不得依赖真实 API key**；graph 直驱行为测在 `tests/test_core/test_loop/test_react_agent.py`（run_graph helper），会话入口行为测在 `tests/test_agent/test_orchestration/test_session.py`；agent 层接口测试用 httpx `ASGITransport` + `app.dependency_overrides`（替换 `build_deps`）
- **扩展位**（纯目录语义，本期未实现）：`agent/orchestration/` 未来多 agent/多会话编排器在此包新增模块；`core/tool/` 未来工具鉴权在此包新增模块；`core/loop/` 未来新循环形态（如 plan-execute）在此包新增模块

## 已知坑（改 core/ai 层前必读）

1. **节点内 `model.astream(messages)` 必须传 config**：节点签名写成 `(state, config)` 并 `astream(messages, config=config)`，否则 bind_tools 的 RunnableBinding 内层模型收不到回调，token 级流式事件整体丢失（退化为一条整段文本）
2. **`core/loop/react_agent.py` 不能加 `from __future__ import annotations`**：langgraph 用本模块 globals 求值 State 注解，字符串化会 NameError（`Annotated` 找不到）——1.2.11 仍是该求值机制
3. **langchain-core 1.6.2 是 pydantic v2 原生模型**：BaseChatModel 子类（如测试 fake）字段必须用 `from pydantic import Field`，用 `langchain_core.pydantic_v1.Field` 会使 `default_factory` 失效
4. `ChatOpenAI` 1.6.0 的字段名是 `model_name` / `openai_api_base` / `openai_api_key`，但 pydantic alias 使 `model=` / `api_key=` / `base_url=` 构造可用（本项目写法不变）。1.6.0 会把 openai 异常包装为多重继承子类（`OpenAIRateLimitError` 等），`isinstance` 对 openai 原异常仍为 True，config.py 的白名单继续有效
5. `BaseChatModel.bind_tools` 是抽象方法（直接 raise），fake 模型必须自己实现（并加 `@override` 标记，接口签名 1.6.2 与 0.3.86 逐字一致）
6. **ToolNode 必须显式 `handle_tool_errors=True`**：1.x 默认只把 `ToolInvocationError` 转错误 ToolMessage，其余工具异常直接上抛；显式 True 恢复 0.6.x 语义（任何工具异常 → 错误 ToolMessage 投喂 LLM）
7. **chunk 合并会把 invalid_tool_calls 误判为合法调用**：langchain-core 1.x 的 `AIMessageChunk.__add__` 会触发 `init_tool_calls` 校验器，它对残缺 args 宽容解析（`parse_partial_json`），导致 invalid_tool_calls 变 tool_calls；checkpointer 序列化往返同理。本项目在 agent_node 合并后**转成普通 AIMessage** 并用**严格 json 解析**从 `tool_call_chunks` 重建合法/非法判定——新代码不要回归成直接返回 AIMessageChunk
8. **invalid 的 id 必须用确定性 `invalid_<aimessage下标>_<条内序号>`，不得用原始 id（call_9 之类）**：`invalid_tool_calls` 经 `_convert_message_to_dict` 序列化为 `tool_calls` 发送给 API，其对应反馈 ToolMessage 的 `tool_call_id` 必须与之匹配，否则 deepseek 校验「每个 tool_call_id 都有 ToolMessage」报 400（`An assistant message with 'tool_calls' must be followed by tool messages...`）。agent_node（react_agent.py）与 repair（repair.py）两条合成路径都必须用同一确定性 id；FakeChatModel 的 invalid 测试也必须走 tool_call_chunks（残缺 args）而非直接放 invalid_tool_calls（后者会被 __add__ 宽容解析误判为合法调用，测不到真实链路）
9. 依赖锁精确版本（requirements.txt）：升级 langgraph / langchain 前先跑全量测试
9. langchain-community / langchain-experimental 已进入 sunset 维护（0.4.2 起 import 打 DeprecationWarning，无害）：本项目的工具链（FileManagementToolkit、terminal 的 ShellTool）仍依赖它们，后续可迁独立集成包

## 约定

- 代码注释、docstring、commit message 用中文（与现有代码一致）
- Python 3.13（`requires-python = ">=3.13"`）：全面使用现代语法——PEP 604 `X | None`、内置泛型、`collections.abc.Callable`、`@override`、PEP 695 `type` 别名语句、dataclass `slots=True`、f-string、pathlib。**唯一例外**：`core/loop/react_agent.py` 禁止 `from __future__ import annotations`（见已知坑 #2）
- 新增 provider：在 `lang_agent/ai/chat/` 下加模块，import 时 `registry.register(...)` 即可，core/agent 零改动
- 新增工具：`lang_agent/core/tool/tool_registry.py` 的 `register_tool(ToolSpec(...))` 注册，本期不做 tool 鉴权；内置演示工具在 `lang_agent.core.tool` 被 import 时注册（裸 `import lang_agent.core` 不触发，core/__init__ 不做转发导出）
- TDD：改行为先加失败测试（`tests/` 里），跑红 → 最小实现 → 跑绿 → 全量回归


## Plugin 模式（2026-09-10）

- `core/plugin/` 实现 PluginBase、按 agent_id 隔离的 PluginRegistry、RunnableCallable 双函数节点、无持久 checkpoint 的内部 hook graph、wrap 调用链、版本发布与工具审批。plugin 不反向 import loop/agent；具体 state/context 仍在 react_agent，避免类型移动引入循环导入。
- 新建 AgentLoop 自动生成只读 agent_id；创建 loop 后按其 id 注册，再 `update_plugin_hooks(snapshot)` 发布。固定主图为 START → before_agent → before_model → agent → after_model →（before_tool → tools → after_tool → before_model / before_model / after_agent → END）。默认 recursion_limit=80。
- 内部 graph `compile(checkpointer=False)`；六个固定 dispatcher 调用本轮 bundle 中的 graph。before/abefore 等方法放入同一 RunnableCallable，仅同步时显式补线程异步适配器；RunnableCallable 原生 ainvoke fallback 不切线程。
- ChatSession 在 initial state 写入 agent_id/run_id/plugin_revision，绑定 context 副本，不修改缓存的 context。动态更新只影响新轮次；重试/审批恢复必须加载原 bundle。插件配置放入 config，注册时复制；历史定义由部署装配重新加载。
- session 的 checkpoint key 是编码后的 `(agent_id, thread_id)`，测试和调用方使用 `session.run_config(thread_id)`。sqlite 装配在同一 DB 的 lang_agent_identity 表保存身份；自定义装配用 AgentLoop.restore。旧已完成会话通过 migrate_legacy_thread 显式复制，源数据保留。
- HookPause 在外层固定节点调用 interrupt；session.resume/resume_stream + `/chat/resume[/stream]` 使用 Command(resume=...)，不追加 HumanMessage。暂停发 interrupt，不发 done/error。无审批的失败任务可省略 answers 恢复。
- repair 区分 after_model/before_tool/tools 的待执行批，不能给待审批调用合成缺失结果；审批恢复不做 repair/compress。普通新轮次 invoke/stream 都执行 repair → compress。
- 模型 wrapper 包裹流式 terminal；最终响应仍规范化为普通 AIMessage，invalid 确定性 id 与 repair 保持一致。主模型 token 带 plugin_model_role=primary，辅助调用不外发；after hook 的最终结果由 checkpoint 塑形。需要审核的插件声明 requires_buffered_output。
- 工具链统一接 ToolNode.awrap_tool_call，旧同步 wrap_tool_call 经桥接放在插件链最内层。默认 handle_tool_errors=True；PluginError 在工具节点出口重新抛出，避免契约错误被吞成业务结果。ToolApprovalPlugin 拒绝时不执行 wrapper 或工具，批准参数在 terminal 再验证。
- 新测试位于 tests/test_core/test_plugin，覆盖身份隔离、双函数、消息增量、审批更新/恢复、sqlite 重启、缓存/同步桥接、迁移和取消；server/CLI 同步测试接口闭环。
