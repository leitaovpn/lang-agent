# AGENTS.md

本文件为在此仓库工作的 AI 编码代理提供指导。

## 项目概览

lang-agent：基于 langgraph 的 lang agent，分三层：

- **ai**（LLM 接入）：`get_llm(model, provider, protocol)` 注册表工厂，映射为 langchain chat model 调用对象
- **core**（agent_loop）：手搓 StateGraph 的完整 ReAct 循环，`AgentLoop.invoke/stream` 统一入口
- **agent**（对外接口）：FastAPI（`/chat`、`/chat/stream` SSE）+ CLI（纯 HTTP 客户端）

依赖方向严格单向 `agent → core → ai`，禁止反向 import。README.md 面向使用者（用法/API/配置），本文件面向在此仓库开发的代理。

## 开发命令

环境用仓库内 venv（`.venv`，Python 3.9.6），**不要用系统 python3**：

```bash
.venv/bin/pip install -r requirements-dev.txt   # 安装依赖（锁精确版本）
.venv/bin/python -m pytest                      # 全部测试（53 个）
.venv/bin/python -m pytest tests/test_core/test_react_agent.py::test_invoke_plain_text   # 单个测试
.venv/bin/python -m lang_agent.agent.cli serve --port 8000          # 起服务
.venv/bin/python -m lang_agent.agent.cli chat --msg "..." --stream  # CLI 冒烟
```

没有 lint/format/CI 配置。IDE 诊断按系统 Python 3.13 的 typeshed 报类型错，以 3.9 运行时实际行为为准。

## 架构要点

- **State**：`AgentState` TypedDict，`messages: Annotated[list[AnyMessage], add_messages]`（`AnyMessage` 来自 langchain_core.messages，`add_messages` 来自 langgraph.graph.message），外加只读元数据字段 `system`、`raw_input`
- **拓扑**：`START → agent（bind_tools + 流式合并 chunk）→ should_continue → tools（ToolNode）/ END / agent（重试）`，`tools → agent` 回环
- **LLM 历史不变式**（`core/repair.py`）：checkpoint 里存储的历史必须满足「AIMessage 的每条 tool_call 后面都有对应 ToolMessage，tool_call_id 在**两个 AIMessage 之间**不重合」。invoke/stream 开始前 `_repair_checkpoint_state` 用 `repair_state_for_checkpoint` 修复并写回——范围是最后一条**带 tool_calls** 的 AIMessage 起（坏段可能被本轮末尾的纯文本 AIMessage 推到前缀）；写回用 RemoveMessage 全删再加回，精确重建顺序（add_messages reducer 会把新消息追加到末尾，直接返回修复列表会打乱顺序）。前缀 tool_call_id 完全不参与去重；段内缺结果合成错误 ToolMessage、重复/孤儿丢弃、重复 id 改名、invalid_tool_calls 合成错误反馈；`should_continue` 看到只有 invalid_tool_calls 的 AIMessage 会带反馈回 agent 重试
- **流式**：`graph.astream(stream_mode=["messages", "updates"])` 双通道——messages 给 token 级文本（metadata 按 `langgraph_node` 过滤），updates 给完整 AIMessage（tool_call）与 ToolMessage（tool_result）；事件分类是 `core/events.py` 的纯函数
- **多轮记忆**：`thread_id` + checkpointer。默认 sqlite（`AsyncSqliteSaver`，重启不丢），`memory` 兜底（`InMemorySaver`）。**sqlite 只能在异步上下文创建**：`await build_checkpointer(config)` 后注入 `AgentLoop(..., checkpointer=...)`，直接构造会抛 ValueError
- **graph 调用重试**（`core/retry.py` + AgentLoopConfig 重试字段）：`ainvoke/astream` 异常后按**瞬时异常白名单**重试（默认 httpx 传输/超时 + `RetryableError` 标记；agent 层 config.py 注入 openai 限流/连接/超时/5xx 类），指数退避（默认 0.5/1/2s，`retry_max_attempts=3`）。**重试用 `input=None` 从 checkpoint 续跑**——失败的超步重执行、输入消息不重复追加（已实测验证）；**续跑前（`_resume_input`）会先执行 `_repair_checkpoint_state` 修复 checkpoint 里的消息**（本轮中途产生的坏段在入口修复之后才写入，不修则续跑时 LLM 拿到未修复历史；已实测 aupdate_state 不会清掉 pending 任务）；首步即失败（无 checkpoint）时复用原输入。确定性错误不重试；耗尽后 invoke 抛异常、stream 出 error 事件。**注意分层**：openai SDK 对 5xx/429 自带约 3 次内部重试，会先吸收掉短故障，我们的层只在 SDK 耗尽后接管（E2E 实测：mock 连挂 5 次，SDK 消耗 4 次后我们的层重试 1 次成功）
- **测试设施**：`tests/` 镜像包结构；LLM 一律用 `tests/conftest.py` 的 `FakeChatModel`（脚本队列驱动，`bind_tools` 返回真实 RunnableBinding），**测试不得依赖真实 API key**；agent 层接口测试用 httpx `ASGITransport` + `app.dependency_overrides`

## 已知坑（改 core/ai 层前必读）

1. **节点内 `model.astream(messages)` 必须传 config**：节点签名写成 `(state, config)` 并 `astream(messages, config=config)`，否则 bind_tools 的 RunnableBinding 内层模型收不到回调，token 级流式事件整体丢失（退化为一条整段文本）
2. **`react_agent.py` 不能加 `from __future__ import annotations`**：langgraph 用本模块 globals 求值 State 注解，字符串化会 NameError（`Annotated` 找不到）
3. **langchain-core 0.3.86 是 pydantic v2 原生模型**：BaseChatModel 子类（如测试 fake）字段必须用 `from pydantic import Field`，用 `langchain_core.pydantic_v1.Field` 会使 `default_factory` 失效
4. `ChatOpenAI` 0.3.35 的字段名是 `model_name` / `openai_api_base` / `openai_api_key`（没有 `model` / `api_key`）
5. `BaseChatModel.bind_tools` 是抽象方法（直接 raise），fake 模型必须自己实现
6. 依赖锁精确版本（requirements.txt）：升级 langgraph / langchain 前先跑全量测试

## 约定

- 代码注释、docstring、commit message 用中文（与现有代码一致）
- Python 3.9 兼容：运行时会被求值的注解（TypedDict 字段等）不用 `X | Y`；普通函数注解可用内置泛型 `list[...]` / `dict[...]`
- 新增 provider：在 `lang_agent/ai/chat/` 下加模块，import 时 `registry.register(...)` 即可，core/agent 零改动
- 新增工具：`lang_agent/core/tool_registry.py` 的 `register_tool(ToolSpec(...))` 注册，本期不做 tool 鉴权
- TDD：改行为先加失败测试（`tests/` 里），跑红 → 最小实现 → 跑绿 → 全量回归
