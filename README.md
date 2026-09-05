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
# Python 3.9+，创建 venv 并安装依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 配置 deepseek API key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx
```

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
```

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

响应 `ChatResponse`：`{"thread_id": "...", "answer": "...", "tool_calls": [...]}`

错误语义：未知 provider / protocol → `400`；缺 API key → `500`（附环境变量名）。

### `POST /chat/stream`

同请求体，SSE 响应（`text/event-stream`），事件序列：

| 事件 | data | 说明 |
|---|---|---|
| `llm_token` | `{"text": "..."}` | agent 逐 token 文本 |
| `tool_call` | `{"id", "name", "arguments"}` | 模型发起工具调用 |
| `tool_result` | `{"tool_call_id", "name", "content"}` | 工具执行结果 |
| `done` | `{"thread_id", "final_text", "tool_calls"}` | 循环正常结束 |
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
START → agent（LLM + bind_tools，流式合并）→ tools_condition
          ├─ 无 tool_calls → END
          └─ 有 tool_calls → tools（ToolNode 执行真实工具）→ agent（回环）
```

统一入口（屏蔽底层 agent 差异）：

```python
from lang_agent.ai import get_llm
from lang_agent.core import AgentLoop

loop = AgentLoop(llm=get_llm(model="deepseek-v4-flash"))

result = await loop.invoke("计算 (3+5)*7", thread_id="t1")       # 一次性拿结果
async for event in loop.stream("计算 (3+5)*7", thread_id="t1"):  # token 级事件流
    print(event.type, event.data)
```

多轮记忆：`thread_id` + checkpointer（默认 SQLite 持久化，服务重启不丢）。

**新增工具**（[lang_agent/core/tool_registry.py](lang_agent/core/tool_registry.py) 注册即可）：

```python
from pydantic import BaseModel, Field
from lang_agent.core.tool_registry import ToolSpec, register_tool

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
.venv/bin/python -m pytest        # 53 个测试，全部用 FakeChatModel 注入，不依赖真实 API key
```

## 已知注意点

- venv Python 3.9.6 + macOS LibreSSL：urllib3 v2 会打印 `NotOpenSSLWarning`，无害
- langgraph 0.6.11 + langchain-core 0.3.86 的组合有几个坑（节点内 config 传递、TypedDict 注解、AsyncSqliteSaver 异步创建等），已在代码注释与 `tests/conftest.py` 中固化写法，改动 core/ai 层前建议先看现有实现
