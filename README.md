# lang-agent

基于 [langgraph](https://github.com/langchain-ai/langgraph) 的 lang agent，分三层：

| 层 | 职责 | 目录 |
|---|---|---|
| **ai** | 对接不同 LLM：通过 `(model, provider, protocol)` 三个参数获取对应的 LLM 调用对象 | [lang_agent/ai/](lang_agent/ai/) |
| **core** | agent_loop 实现：屏蔽底层 agent 差异，完整 ReAct 循环（LLM 调用 + tool_call），支持插件生命周期与工具审批 | [lang_agent/core/](lang_agent/core/) |
| **agent** | FastAPI 对外暴露接口 + CLI 命令调用 API | [lang_agent/agent/](lang_agent/agent/) |

依赖方向严格单向：`agent → core → ai`。

## 快速开始

```bash
# Python 3.13+，创建 venv 并安装依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 配置 deepseek API key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx
```

项目提供 `.envrc`，可通过 [direnv](https://direnv.net/) 在进入目录时自动激活 `.venv`，并将 Python 字节码缓存集中写入 `.cache/pycache`。以 macOS + zsh 为例：

```bash
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
source ~/.zshrc
direnv allow
```

配置生效后可以直接使用 `python`、`pip`、`pytest` 等命令；不使用 direnv 时，继续通过 `.venv/bin/...` 显式调用即可。`.cache/`、`__pycache__/` 和 `.venv/` 均已加入 Git 忽略规则。

启动服务：

```bash
.venv/bin/python -m lang_agent.agent.cli serve            # 默认 127.0.0.1:8000
```

CLI 调用（纯 HTTP 客户端，走本地 API）：

```bash
# 同步
.venv/bin/python -m lang_agent.agent.cli chat --msg "你好"

# 流式（token 级打印，工具过程灰显）
.venv/bin/python -m lang_agent.agent.cli chat --msg "计算 (3+5)*7" --stream

# 多轮对话（thread_id 记忆历史）
.venv/bin/python -m lang_agent.agent.cli chat --msg "计算 (3+5)*7" --stream --thread-id my-thread
.venv/bin/python -m lang_agent.agent.cli chat --msg "再乘 2 是多少" --stream --thread-id my-thread

# 交互模式（默认流式、多轮上下文连续；无服务端时自动拉起，端口被占自动换端口）
.venv/bin/python -m lang_agent.agent.cli chat
```

交互模式内支持 `/help`、`/exit`、`/quit`；Ctrl+C 中断当前生成或退出。
工具结果默认截断到 500 字符再打印（`--max-output-chars` 可覆盖，`<=0` 不截断）；`--thread-id` 可续指定会话。

直接调 API：

```bash
# 同步
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "计算 (3+5)*7", "thread_id": "t1"}'

# SSE 流式
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "计算 (3+5)*7", "thread_id": "t1"}'
```

## API

### `POST /chat`

请求体 `ChatRequest`：

```json
{
  "model": "deepseek-v4-flash",   // 默认 deepseek-v4-flash
  "provider": "deepseek",         // 默认 deepseek
  "protocol": "chat_response",    // 默认 chat_response（OpenAI 兼容 chat 接口）
  "thread_id": "default",         // 多轮对话线程 id
  "message": "计算 (3+5)*7",
  "system": null                  // 可选系统提示
}
```

响应 `ChatResponse`：`{"agent_id": "...", "thread_id": "...", "answer": "...", "tool_calls": [...], "status": "completed", "interrupts": []}`

错误语义：未知 provider / protocol → `400`；缺 API key → `500`（附环境变量名）。

### `POST /chat/stream`

同请求体，SSE 响应（`text/event-stream`），事件序列：

| 事件 | data | 说明 |
|---|---|---|
| `thinking_token` | `{"text": "..."}` | 模型逐 token 思考（reasoning_content） |
| `llm_token` | `{"text": "..."}` | agent 逐 token 文本 |
| `tool_call` | `{"id", "name", "arguments"}` | 模型发起工具调用 |
| `tool_result` | `{"tool_call_id", "name", "content"}` | 工具执行结果 |
| `interrupt` | `{"agent_id", "thread_id", "interrupts"}` | 等待审批，此次流结束 |
| `done` | `{"agent_id", "thread_id", "final_text", "tool_calls"}` | 循环正常结束 |
| `error` | `{"message"}` | 循环异常终止 |

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | deepseek 的 API key（必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 覆盖 deepseek base_url |
| `LANG_AGENT_MODEL` | `deepseek-v4-flash` | 默认模型 |
| `LANG_AGENT_PROVIDER` | `deepseek` | 默认 provider |
| `LANG_AGENT_PROTOCOL` | `chat_response` | 默认 protocol |
| `LANG_AGENT_CHECKPOINTER` | `sqlite` | 多轮记忆存储：`sqlite`（重启不丢）或 `memory`（进程内） |
| `LANG_AGENT_DB_PATH` | `~/.lang-agent/checkpoints.sqlite` | checkpoint 数据库路径 |
| `LANG_AGENT_HOST` / `LANG_AGENT_PORT` | `127.0.0.1` / `8000` | 服务地址 |

## 架构

### ai 层：`get_llm(model, provider, protocol)`

注册表工厂，把三个参数映射为 langchain chat model 调用对象：

```python
from lang_agent.ai import get_llm

llm = get_llm(model="deepseek-v4-flash", provider="deepseek", protocol="chat_response")
```

- `provider` → base_url、API key 环境变量（`ProviderConfig`）
- `protocol` → 接入风格（`chat_response` = OpenAI 兼容 chat 接口；未来可加 anthropic 风格）
- 异常：`UnknownProviderError` / `UnknownProtocolError` / `MissingApiKeyError`

**新增 provider**：在 [lang_agent/ai/chat/](lang_agent/ai/chat/) 下加一个模块，import 时注册即可，core/agent 层零改动：

```python
from langchain_openai import ChatOpenAI
from lang_agent.ai.base import ProviderConfig
from lang_agent.ai.registry import registry

registry.register(
    provider="my_provider",
    protocols={"chat_response": lambda *, model, api_key, base_url, **kw:
               ChatOpenAI(model=model, api_key=api_key, base_url=base_url, **kw)},
    config=ProviderConfig(name="my_provider", base_url="https://...", api_key_env="MY_API_KEY"),
)
```

### core 层：`AgentLoop`

手搓 StateGraph 的完整 ReAct 循环：

```
START → before_agent → before_model → agent → after_model
          ├─ 完成 → after_agent → END
          ├─ invalid/重试 → before_model
          └─ 工具调用 → before_tool → tools → after_tool → before_model
```

`AgentLoop` 是纯 graph 薄封装：构造只编译 graph，invoke/stream 与 `graph.ainvoke/astream` 同形透传，不持有 LLM/工具——每次 run 经 `context=` 注入：

```python
from langchain_core.messages import HumanMessage
from lang_agent.ai import get_llm
from lang_agent.core.loop import AgentContext, AgentLoop, AgentLoopConfig

loop = AgentLoop(config=AgentLoopConfig())   # checkpointer 默认 memory
context = AgentContext(llm=get_llm(model="deepseek-v4-flash"), tools=[])

cfg = {"configurable": {"thread_id": "t1"}}
result = await loop.invoke({"messages": [HumanMessage(content="计算 (3+5)*7")]}, config=cfg, context=context)
async for chunk in loop.stream({"messages": [HumanMessage(content="继续")]}, config=cfg, context=context):
    print(chunk)   # 原始事件；结果塑形/事件分类由 agent 层 ChatSession 提供
```

多轮记忆：`thread_id` + checkpointer（memory 或 sqlite 持久化，见 `AgentLoopConfig.checkpointer_kind`）。

**新增工具**（[lang_agent/core/tool/tool_registry.py](lang_agent/core/tool/tool_registry.py) 注册即可）：

```python
from pydantic import BaseModel, Field
from lang_agent.core.tool import ToolSpec, register_tool

class Args(BaseModel):
    city: str = Field(description="城市名")

def weather(city: str) -> str:
    return f"{city} 今天晴"

register_tool(ToolSpec(name="weather", description="查询天气", fn=weather, args_schema=Args))
```

内置演示工具：`calculator`（ast 白名单安全求值）、`string_reverse`、`string_len`。

## 测试

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest        # 全量测试，使用 FakeChatModel 注入，不依赖真实 API key
```

## 已知注意点

- langgraph 1.2.11 + langchain-core 1.6.2 的组合有几个坑（节点内 config 传递、TypedDict 注解、AsyncSqliteSaver 异步创建、ToolNode handle_tool_errors、chunk 合并对 invalid_tool_calls 的误判等），已在代码注释与 `tests/conftest.py` 中固化写法，改动 core/ai 层前建议先看现有实现
- langchain-community / langchain-experimental 已进入 sunset 维护，import 会打 DeprecationWarning，无害

## 插件与 agent 身份

每个新 `AgentLoop` 自动生成只读 `agent_id`。先创建 loop，再按其 ID 注册和发布插件；不同 agent 可以注册同名插件，父插件依赖只在同一个 agent 内解析。

```python
from langchain_core.messages import SystemMessage
from lang_agent.agent.orchestration import ChatSession
from lang_agent.core.loop import AgentContext, AgentLoop
from lang_agent.core.plugin import PluginBase, PluginRegistry

class PromptPlugin(PluginBase):
    name = "prompt"
    version = "1"
    config = {"prompt": "请用中文简洁回答。"}

    async def awrap_model_hook(self, request, handler):
        return await handler(request.override(
            messages=[SystemMessage(content=self.config["prompt"]), *request.messages],
        ))

loop = AgentLoop()
registry = PluginRegistry()
registry.register(PromptPlugin(), agent_id=loop.agent_id, hooks=["wrap_model_hook"])
loop.update_plugin_hooks(registry.snapshot(agent_id=loop.agent_id))
session = ChatSession(loop=loop)
result = await session.invoke("你好", thread_id="t1", context=AgentContext(llm=llm, tools=[]))
```

六组 node hook 为 `before_agent/after_agent/before_model/after_model/before_tool/after_tool`；同步方法和对应的 `a...` 异步方法放入同一个 RunnableCallable 节点。异步运行选择异步实现，仅同步实现在线程中执行。node hook 返回 `HookResult(update=...)`，消息修改用同 id 替换或 `RemoveMessage`，插件自己的持久数据放在 `plugin_state[插件名]`。

模型与工具分别由 `wrap_model_hook/awrap_model_hook`、`wrap_tool_hook/awrap_tool_hook` 拦截。父依赖和注册顺序决定 node hook 执行顺序（after 也是正序）；wrapper 请求正序、响应逆序。同步 wrapper 可通过同步 handler 调用仅异步工具，嵌套上限为 8 层；每层 handler 调用上限为 3 次。有副作用的 hook/工具需自行保证重放幂等。

`registry.replace/unregister/snapshot` 也需要 `agent_id`。修改注册表后调用 `loop.update_plugin_hooks(...)` 才生效；主图不重编译，每一轮及其重试/审批恢复绑定原插件版本，新版本从下一轮开始使用。配置变化应写入 `config` 并发布新实例/版本，不要把请求数据放进共享插件属性。

使用 `loop.describe_plugin_hooks()` 查看身份、版本和各 hook 的绑定。自定义模型/工具节点默认只享有 node hook；显式声明 `node_wrap_hooks=frozenset({"wrap_model_hook"})` 后，节点实现还须调用默认节点工厂所提供的模型调用流程，才能执行对应 wrapper。

`ChatSession` 将 `(agent_id, thread_id)` 编码成内部 checkpoint key。手动读取 checkpoint 使用 `session.run_config("t1")`，不要使用裸 thread_id。直接调用 core graph 时，需自行使用隔离的 config，并通过 `loop.bind_plugin_context(context)` 及 initial state 的 `agent_id/plugin_revision/run_id` 固定本轮版本；invoke/stream 保持参数原样透传。

### 工具审批与恢复

在相同注册表中增加 `ToolApprovalPlugin`：

```python
from lang_agent.core.plugin.approval import ToolApprovalPlugin

registry.register(ToolApprovalPlugin(), agent_id=loop.agent_id, hooks=["before_tool"])
loop.update_plugin_hooks(registry.snapshot(agent_id=loop.agent_id))
```

`POST /chat` 响应新增 `agent_id`、`status`、`interrupts`。`status="interrupted"` 表示等待审批；SSE 使用 `interrupt` 事件，此时没有 `done`。批准、拒绝或编辑参数后恢复：

```json
{
  "agent_id": "服务器返回的 agent_id",
  "thread_id": "t1",
  "answers": {
    "服务器返回的 interrupt id": "approve"
  }
}
```

将上述请求发送到 `POST /chat/resume` 或 `/chat/resume/stream`。答案也可以是 `"reject"`，或 `{"action":"edit","args":{"expression":"2+2"}}`。一个批次有多个工具时，可能逐个产生新的审批请求；拒绝直接返回错误 ToolMessage，真实工具与其 wrapper 都不会运行。自定义 `HookPause` 可以定义自己的 JSON 答案语义，覆盖 `PluginBase.validate_answer(key, payload, answer)` 在提交恢复命令前校验；错误答案不会消耗待审批状态。

CLI 在交互终端中询问答案；脚本模式打印暂停信息并以退出码 3 结束，保留审批：

```bash
.venv/bin/python -m lang_agent.agent.cli chat --resume \
  --agent-id <agent_id> --thread-id t1 --answers '{"<interrupt_id>":"approve"}'
```

程序内对应 `session.resume(thread_id=..., answers=..., context=...)` / `resume_stream(...)`。普通失败任务可省略 answers 从 checkpoint 续跑。未完成线程不能直接追加新用户消息；身份或审批状态不匹配返回 HTTP 409。

token 是暂定输出，`done.final_text` 是 after hook 完成后的最终文本。CLI 会在两者不同时打印校正后的回答。需要禁止未审核内容外发的插件可声明 `requires_buffered_output=True`，session 将关闭该轮原始 token 转发，完成后发送最终文本。

### 持久身份与旧会话迁移

默认 sqlite 服务将 agent 身份保存在同一数据库中，重启后复用。自行装配时保存 `loop.agent_id`，用 `AgentLoop.restore(agent_id=..., checkpointer=...)` 恢复，并按原版本重新注册插件；旧代码/配置未加载时恢复明确失败，不自动切到新版。checkpoint 不保存可执行插件对象，部署方需保留历史插件版本和配置；可通过 `registry.snapshot(agent_id=...).manifest()` 导出恢复定义，配置中不要包含凭据。

升级前的裸 thread_id 历史不会自动混入新的 agent。只对已完成的旧会话显式执行 `await session.migrate_legacy_thread("t1")`，复制历史到当前 agent，保留原数据；目标已存在或源线程未完成时拒绝迁移。首次升级前应先完成旧版的待执行任务。

主图新增 hook 节点后，`AgentLoopConfig.recursion_limit` 默认从 25 调整为 80，仍表示主图步数上限。服务仍是本地应用接口，agent_id 是路由/隔离标识，不是用户认证凭据；多用户部署需由宿主接入身份认证和 agent 访问授权。
