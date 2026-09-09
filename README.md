# lang-agent

基于 [langgraph](https://github.com/langchain-ai/langgraph) 的 lang agent，分三层：

| 层 | 职责 | 目录 |
|---|---|---|
| **ai** | 对接不同 LLM：通过 `(model, provider, protocol)` 三个参数获取对应的 LLM 调用对象 | [lang_agent/ai/](lang_agent/ai/) |
| **core** | agent_loop 实现：屏蔽底层 agent 差异，完整 ReAct 循环（LLM 调用 + tool_call），暂不做 tool 鉴权 | [lang_agent/core/](lang_agent/core/) |
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

# 重新处理某线程尚未提交的工具审批
.venv/bin/python -m lang_agent.agent.cli chat --resume --thread-id my-thread
```

交互模式内支持 `/help`、`/exit`、`/quit`；Ctrl+C 中断当前生成或退出。
所有工具调用默认先显示名称与参数并逐条询问 `允许执行？[y/N]`，回车表示拒绝；审批提交前工具不会执行。工具结果默认截断到 500 字符再打印（`--max-output-chars` 可覆盖，`<=0` 不截断）；`--thread-id` 可续指定会话。

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

响应 `ChatResponse`。正常完成时 `status` 为 `completed`；等待审批时为 `awaiting_approval`，并通过 `approval.tool_calls` 返回待审批调用：

```json
{
  "thread_id": "t1",
  "answer": "",
  "tool_calls": [{"id": "call_1", "name": "calculator", "arguments": {"expression": "1+1"}}],
  "status": "awaiting_approval",
  "approval": {
    "approval_id": "approval_...",
    "tool_calls": [{"id": "call_1", "name": "calculator", "arguments": {"expression": "1+1"}}]
  }
}
```

错误语义：未知 provider / protocol → `400`；缺 API key → `500`（附环境变量名）。

### `POST /chat/stream`

同请求体，SSE 响应（`text/event-stream`），事件序列：

| 事件 | data | 说明 |
|---|---|---|
| `thinking_token` | `{"text": "..."}` | 模型逐 token 思考（reasoning_content） |
| `llm_token` | `{"text": "..."}` | agent 逐 token 文本 |
| `tool_call` | `{"id", "name", "arguments"}` | 模型发起工具调用 |
| `approval_required` | `{"approval_id", "tool_calls"}` | graph 已暂停，等待人工审批；本次 SSE 随后结束 |
| `tool_result` | `{"tool_call_id", "name", "content"}` | 工具执行结果 |
| `done` | `{"thread_id", "final_text", "tool_calls"}` | 循环正常结束 |
| `error` | `{"message"}` | 循环异常终止 |

### 工具审批与恢复

把 `POST /chat` 返回的 `approval_id` 和每条调用的决定提交到 `POST /chat/resume`；流式客户端使用 `POST /chat/resume/stream`。决定必须完整覆盖当前批次，动作只能是 `approve` 或 `reject`：

```json
{
  "thread_id": "t1",
  "approval_id": "approval_...",
  "decisions": [
    {"tool_call_id": "call_1", "action": "approve"}
  ]
}
```

`GET /chat/approval?thread_id=t1` 可查询尚未处理的批次。新消息遇到待审批线程返回 `409`；不存在的审批返回 `404`；过期或不完整的决定返回 `400`。拒绝的工具不会执行，graph 会生成错误 `ToolMessage` 供模型继续回答。

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
START → agent（LLM + bind_tools，流式合并）→ should_continue
          ├─ 无 tool_calls → END
          └─ 有 tool_calls → approval（可中断恢复）→ tools（仅执行获批工具）→ agent（回环）
```

`AgentLoop` 是纯 graph 薄封装：构造只编译 graph，invoke/stream 与 `graph.ainvoke/astream` 同形透传，不持有 LLM/工具——每次 run 经 `context=` 注入：

```python
from lang_agent.ai import get_llm
from lang_agent.core.loop import (
    AgentContext,
    AgentLoop,
    AgentLoopConfig,
    require_human_approval,
)

loop = AgentLoop(
    config=AgentLoopConfig(tool_approval_hook=require_human_approval)
)  # checkpointer 默认 memory；钩子通过 config 注入
context = AgentContext(llm=get_llm(model="deepseek-v4-flash"), tools=[])

cfg = {"configurable": {"thread_id": "t1"}}
result = await loop.invoke("计算 (3+5)*7", config=cfg, context=context)
async for chunk in loop.stream("计算 (3+5)*7", config=cfg, context=context):
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
.venv/bin/python -m pytest        # 全部使用 FakeChatModel 注入，不依赖真实 API key
```

## 已知注意点

- langgraph 1.2.11 + langchain-core 1.6.2 的组合有几个坑（节点内 config 传递、TypedDict 注解、AsyncSqliteSaver 异步创建、ToolNode handle_tool_errors、chunk 合并对 invalid_tool_calls 的误判等），已在代码注释与 `tests/conftest.py` 中固化写法，改动 core/ai 层前建议先看现有实现
- langchain-community / langchain-experimental 已进入 sunset 维护，import 会打 DeprecationWarning，无害
