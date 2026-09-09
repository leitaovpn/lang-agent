# 设计:core / agent 两层模块化重组

日期:2026-09-09
状态:已批准(2026-09-09)

## 背景

当前 `core/` 与 `agent/` 两层是扁平文件结构,文件已开始承载多职责(react_agent.py 357 行、session.py 287 行)。本次重组把两层切成职责清晰的子包,并为未来能力预留扩展位。

## 目标

1. core 层拆出 `loop`(ReAct 循环 + 生命周期支撑)与 `tool`(工具实现 + 注册表)两个子包
2. agent 层拆出 `orchestration`(会话编排)、`server`(FastAPI 对外接口)、`cli`(HTTP 客户端)三个子包
3. 纯重组:代码逻辑零改动,108 个测试全绿
4. 彻底切换 import 路径:旧扁平路径(如 `lang_agent.core.react_agent`)全部废弃,不保留兼容转发

## 非目标(本期不做)

- 多 agent / 多会话编排、tool 鉴权、新 loop 形态——只预留目录语义,在 AGENTS.md 标注位置,不写占位代码/抽象基类
- ai 层不动(已有 chat/ 子包结构,不在本次范围)

## 目标结构

```
lang_agent/
├── ai/                            # 不动
├── core/
│   ├── __init__.py                # 仅 docstring(旧扁平路径彻底废弃)
│   ├── loop/                      # ReAct 循环 + 全部生命周期支撑
│   │   ├── __init__.py            # 导出 AgentLoop/AgentContext/AgentLoopConfig/ReActNode/
│   │   │                          #   build_checkpointer/build_default_agent_node/build_default_tools_node/
│   │   │                          #   AgentEvent/ConversationResult
│   │   ├── react_agent.py         # 原 core/react_agent.py
│   │   ├── events.py              # 原 core/events.py
│   │   ├── repair.py              # 原 core/repair.py
│   │   ├── retry.py               # 原 core/retry.py
│   │   └── compress.py            # 原 core/compress.py
│   └── tool/
│       ├── __init__.py            # 导出 register_tool/get_tool/instantiate_tools/ToolSpec
│       ├── tools.py               # 原 core/tools.py
│       └── tool_registry.py       # 原 core/tool_registry.py
└── agent/
    ├── __init__.py                # 仅 docstring
    ├── orchestration/             # 编排:会话级 repair/compress/retry/结果塑形/事件分类
    │   ├── __init__.py            # 导出 ChatSession
    │   ├── session.py             # 原 agent/session.py
    │   ├── config.py              # 原 agent/config.py
    │   ├── deps.py                # 原 agent/deps.py
    │   └── schemas.py             # 原 agent/schemas.py
    ├── server/
    │   ├── __init__.py            # 导出 app(uvicorn "lang_agent.agent.server:app" 保持可用)
    │   └── app.py                 # 原 agent/server.py 改名
    └── cli/
        ├── __init__.py            # 仅 docstring
        ├── main.py                # 原 agent/cli.py 改名
        └── __main__.py            # python -m lang_agent.agent.cli 入口保持可用
```

## import 改写清单(全部为路径调整,无逻辑改动)

| 文件 | 改写 |
|---|---|
| core/loop/react_agent.py | `from lang_agent.core.compress import truncate_tool_outputs` → 相对导入 `.compress`;`from lang_agent.core.repair import INVALID_ID_PREFIX` → `.repair` |
| core/tool/tool_registry.py | `from lang_agent.core import tools as builtin_tools` → `from . import tools` |
| agent/orchestration/session.py | `from lang_agent.core import ...` → `from lang_agent.core.loop import ...`;`core.compress/events/react_agent/repair/retry` → `core.loop.<同名>` |
| agent/orchestration/config.py | `from lang_agent.core import AgentLoopConfig` → `from lang_agent.core.loop import AgentLoopConfig` |
| agent/orchestration/deps.py | `from lang_agent.agent import config as app_config` → 包内相对 `.config`;`agent.session` → `.session`;core 路径 → `core.loop` / `core.tool` |
| agent/server/app.py | `lang_agent.agent.deps/schemas` → `lang_agent.agent.orchestration.deps/schemas` |
| agent/cli/main.py | `lang_agent.agent.config` → `lang_agent.agent.orchestration.config` |

各子包内部其余交叉引用(如 loop 内 react_agent ↔ compress/repair)一律改相对导入。

注意:react_agent.py 移动后仍**禁止** `from __future__ import annotations`(AGENTS.md 已知坑 #2,langgraph 用模块 globals 求值 State 注解)。

## 入口保持可用

- `python -m lang_agent.agent.cli serve|chat`:`cli/__main__.py` 转发到 `main.py` 的 `main()`
- uvicorn 字符串 `lang_agent.agent.server:app`:`server/__init__.py` 转发导出,`main.py` 的 `_serve` 里字符串无需改

## 依赖方向(不变)

```
agent(orchestration ← server/cli) → core(loop/tool) → ai
```

- server → orchestration(deps/schemas)
- cli → orchestration(仅 DEFAULT_HOST/DEFAULT_PORT 常量)+ 运行时 uvicorn 字符串引用 server(惰性 import)
- 无环,禁止反向

## 测试重组

`tests/` 镜像新包结构:

```
tests/
├── conftest.py                    # 不动(不含 lang_agent 引用)
├── test_ai/                       # 不动
├── test_core/
│   ├── test_loop/
│   │   ├── test_react_agent.py
│   │   ├── test_events.py
│   │   ├── test_repair.py
│   │   ├── test_retry.py
│   │   └── test_compress.py
│   └── test_tool/
│       ├── test_tools.py
│       └── test_tool_registry.py
└── test_agent/
    ├── test_orchestration/
    │   ├── test_session.py
    │   ├── test_deps.py
    │   └── test_schemas.py
    ├── test_server/
    │   └── test_server.py
    └── test_cli/
        └── test_cli.py
```

所有测试文件的 import 同步切换新路径。测试内容与断言零改动(纯重组)。

## 文档同步

- **AGENTS.md**:「架构要点」「已知坑」「开发命令」里的旧路径(`core/react_agent.py`、`agent/session.py` 等)更新为新路径;增加「扩展位」标注(orchestration 未来多 agent 编排器、core/tool 未来鉴权、core/loop 未来新循环形态)
- **README.md**:代码示例改新路径——`from lang_agent.core import AgentLoop` → `from lang_agent.core.loop import AgentLoop`;`from lang_agent.core.tool_registry import ToolSpec, register_tool` → `from lang_agent.core.tool import ToolSpec, register_tool`;CLI 命令(`python -m lang_agent.agent.cli ...`)不变
- **CLAUDE.md**：仅引用 AGENTS.md 与命令，虚拟环境命名问题不在本次范围

## 验证方式

1. `.venv/bin/ruff check lang_agent/ tests/` 通过
2. `.venv/bin/mypy lang_agent tests --explicit-package-bases` 通过
3. `.venv/bin/python -m pytest` 108 个测试全绿
4. CLI 冒烟:端到端用 /tmp 临时 OpenAI 兼容 mock 服务 + `DEEPSEEK_BASE_URL` 指向它(CLAUDE.md 记录的历史做法),跑 `python -m lang_agent.agent.cli chat --msg "计算 (3+5)*7" --stream`
5. 提交前确认无旧路径残留:`grep -rn "lang_agent.core.react_agent\|lang_agent.core.events\|lang_agent.core.compress\|lang_agent.core.repair\|lang_agent.core.retry\|lang_agent.core.tools\|lang_agent.core.tool_registry\|lang_agent.agent.session\|lang_agent.agent.deps\|lang_agent.agent.schemas\|lang_agent.agent.config" lang_agent/ tests/ README.md AGENTS.md` 应为空(`lang_agent.agent.server`/`lang_agent.agent.cli` 是新包路径,uvicorn 字符串与 `-m` 入口引用属预期保留)

## 已确认决策(问答记录)

1. 调整性质:重组 + 预留新能力(本期不实现)
2. core 切分:子包目录,支撑逻辑(repair/retry/compress/events)归入 loop/
3. agent 切分:orchestration 收揽 session/config/deps/schemas
4. 兼容策略:彻底切换,旧扁平路径全部废弃
5. 扩展位形态:纯目录语义 + AGENTS.md 文档标注,不写占位代码
