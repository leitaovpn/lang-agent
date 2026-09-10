# Plugin 二次修订（移除节点注入 + 包上移）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 plugin.md 第 15 节修订——移除 AgentLoop 自定义节点注入，并把 plugin 包从 `lang_agent/core/plugin/` 上移到 `lang_agent/plugin/`。

**Architecture:** 两个独立提交：先纯移动（git mv 保留 rename 追踪 + 全量 import 路径更新），再删注入（构造参数、能力校验、相关测试与文档）。每个提交独立通过完整门禁。

**Tech Stack:** Python 3.13、langgraph 1.2.11、pytest（asyncio_mode=auto）、ruff、mypy 双版（主 venv 2.3.1 + `.venv-check` 1.18.2）。

## Global Constraints

- 所有 Python 命令用 `.venv/bin/...`；完整门禁命令：`.venv/bin/ruff check lang_agent/ tests/`、`.venv/bin/mypy lang_agent tests --explicit-package-bases`、`.venv-check/bin/mypy --python-executable .venv/bin/python lang_agent tests --explicit-package-bases`、`.venv/bin/python -m pytest -q`（四个全绿才算过）
- `react_agent.py` 禁止 `from __future__ import annotations`（langgraph 运行时求值注解依赖模块 globals）
- plugin 包不得 import loop/agent（依赖方向 loop → plugin 单向）
- 提交信息用中文 conventional-commit 风格（与仓库既有提交一致）
- 当前分支 `feature/plugin`，直接在分支上提交

---

### Task 1: plugin 包与测试目录上移（纯移动）

**Files:**
- Move: `lang_agent/core/plugin/` → `lang_agent/plugin/`（`git mv`，9 个 .py 文件）
- Move: `tests/test_core/test_plugin/` → `tests/test_plugin/`（`git mv`）
- Modify: `lang_agent/core/loop/react_agent.py`（7 处 import）
- Modify: `lang_agent/agent/orchestration/session.py:43`（1 处 import）
- Modify: `tests/test_agent/test_server/test_server.py:113-114`（2 处 import）
- Modify: `tests/test_plugin/test_plugins.py`（12 处 import）
- Modify: `lang_agent/__init__.py`（分层 docstring）

**Interfaces:**
- Consumes: 无（本任务是纯移动）
- Produces: 包位置 `lang_agent/plugin`（`__init__.py` 导出不变：PluginBase、PluginRegistry、PluginSpecSnapshot、HookPause/HookResult/HookRuntime、ModelRequest/ModelResponse、PluginError/PluginAgentMismatch/PluginRevisionUnavailable）；子模块路径 `lang_agent.plugin.{graph,runtime,types,wrappers,approval,tool_validation,registry,base}`

- [ ] **Step 1: git mv 两个目录**

```bash
git mv lang_agent/core/plugin lang_agent/plugin
git mv tests/test_core/test_plugin tests/test_plugin
```

- [ ] **Step 2: 更新 react_agent.py 的 7 处 import**

将以下 import 的 `lang_agent.core.plugin` 全部替换为 `lang_agent.plugin`（sed 即可）：

```bash
sed -i '' 's/lang_agent\.core\.plugin/lang_agent.plugin/g' lang_agent/core/loop/react_agent.py
```

覆盖行：51、57、58、59、60（顶部）与 322、323（函数内 lazy import，含 `enforce_approval` 与 `validate_tool_result`）。改完后 `grep -n "core.plugin" lang_agent/core/loop/react_agent.py` 应为空。

- [ ] **Step 3: 更新其余文件的 import**

```bash
sed -i '' 's/lang_agent\.core\.plugin/lang_agent.plugin/g' \
  lang_agent/agent/orchestration/session.py \
  tests/test_agent/test_server/test_server.py \
  tests/test_plugin/test_plugins.py
```

改完后全仓验证：

```bash
grep -rn "core\.plugin" lang_agent/ tests/ --include="*.py"   # 应为空
grep -rln "lang_agent.plugin" lang_agent/ tests/ --include="*.py"  # 应有 4 个文件
```

- [ ] **Step 4: 更新 lang_agent/__init__.py 分层说明**

当前 docstring：`"""lang-agent：基于 langgraph 的 lang agent。\n\n分层：agent（FastAPI/CLI）→ core（ReAct agent loop）→ ai（LLM 接入）。\n"""`

改为：

```python
"""lang-agent：基于 langgraph 的 lang agent。

分层：agent（FastAPI/CLI）→ core（ReAct agent loop）→ ai（LLM 接入）；
plugin（横向插件协议包，core.loop 与 agent 层共用，不反向导入分层）。
"""
```

- [ ] **Step 5: 运行完整门禁**

```bash
.venv/bin/ruff check lang_agent/ tests/
.venv/bin/mypy lang_agent tests --explicit-package-bases
.venv-check/bin/mypy --python-executable .venv/bin/python lang_agent tests --explicit-package-bases
.venv/bin/python -m pytest -q
```

Expected：ruff 通过、两个 mypy 均 `Success: no issues found in 52 source files`、`159 passed`。

- [ ] **Step 6: 确认 rename 追踪后提交**

```bash
git status --short | head -20   # 应显示 R（rename）行，无新增/删除的 D+A 配对
git add -A
git commit -m "refactor: plugin 包上移到 lang_agent/plugin（纯移动 + import 更新）"
```

---

### Task 2: 移除 AgentLoop 节点注入

**Files:**
- Modify: `lang_agent/core/loop/react_agent.py`（构造签名、属性、update_plugin_hooks）
- Modify: `tests/test_core/test_loop/test_react_agent.py`（make_loop helper、删 2 个注入测试、加 1 个拒绝测试）
- Modify: `AGENTS.md`（AgentLoop 契约行 + Plugin 模式段路径）

**Interfaces:**
- Consumes: Task 1 的 `lang_agent.plugin` 路径
- Produces: `AgentLoop(*, checkpointer=None, config=None)`——不再接受 `agent_node`/`tools_node`/`node_wrap_hooks`；`ReActNode`、`build_default_agent_node`、`build_default_tools_node` 继续从 `lang_agent.core.loop` 导出

- [ ] **Step 1: 写失败测试（注入被拒绝）**

在 `tests/test_core/test_loop/test_react_agent.py` 的 `test_injected_agent_node_is_used` 位置替换为：

```python
async def test_constructor_rejects_node_injection():
    with pytest.raises(TypeError, match="agent_node"):
        AgentLoop(agent_node=lambda *a: None)
    with pytest.raises(TypeError, match="tools_node"):
        AgentLoop(tools_node=lambda *a: None)
```

同时删除 `test_injected_tools_node_is_used`（约 269-280 行）。

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_core/test_loop/test_react_agent.py::test_constructor_rejects_node_injection -q`
Expected: FAIL——`TypeError` 未触发（构造器仍接受注入参数）

- [ ] **Step 3: 修改 AgentLoop 构造签名**

`lang_agent/core/loop/react_agent.py` 约 411-419 行：

```python
    def __init__(
        self,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
        config: AgentLoopConfig | None = None,
    ) -> None:
        self._agent_id = uuid4().hex
        self._plugins = PluginRuntime(self._agent_id)
        self._config = config or AgentLoopConfig()
```

删除紧随其后的三行：`self._custom_agent = agent_node is not None`、`self._custom_tools = tools_node is not None`、`if node_wrap_hooks - {...}: raise PluginError(...)`、`self._node_wrap_hooks = node_wrap_hooks`。

- [ ] **Step 4: 固定使用默认节点工厂**

约 439-444 行，`self._graph = self._build_graph(agent_node or build_default_agent_node(...), tools_node or build_default_tools_node())` 改为：

```python
        # 固定使用内置默认节点：自定义节点会旁路 wrap hook 的执行保证，
        # 不支持注入（见 docs/superpowers/specs/plugin.md 第 15 节）。
        # 默认节点在构造时捕获 config 的截断参数（构造后改配置不生效）。
        self._graph = self._build_graph(
            build_default_agent_node(self._config.compress_tool_output_max_chars),
            build_default_tools_node(),
        )
```

- [ ] **Step 5: 删除 update_plugin_hooks 的能力校验**

约 532-548 行，`update_plugin_hooks` 改为：

```python
    def update_plugin_hooks(
        self, snapshot: PluginSpecSnapshot, *, expected_revision: str | None = None
    ):
        return self._plugins.update(snapshot, expected_revision)
```

- [ ] **Step 6: 更新 make_loop helper**

`tests/test_core/test_loop/test_react_agent.py` 约 26-34 行：

```python
def make_loop(script, *, checkpointer=None, config=None):
    llm = FakeChatModel(responses=list(script))
    loop = AgentLoop(
        config=config,
        checkpointer=checkpointer,
    )
    return loop, llm
```

- [ ] **Step 7: 运行测试确认通过**

```bash
.venv/bin/python -m pytest tests/test_core/test_loop/test_react_agent.py -q
```

Expected：全部通过（含新的 `test_constructor_rejects_node_injection`）

- [ ] **Step 8: 更新 AGENTS.md**

约 52 行，「AgentLoop 薄封装契约」的入参描述改为：

```markdown
- **AgentLoop 薄封装契约**（core/loop/react_agent.py）：构造只编译 graph——入参 `checkpointer` / `config`；节点固定用模块级工厂 `build_default_agent_node(compress_tool_output_max_chars)` / `build_default_tools_node()`（签名 `ReActNode`），不支持注入自定义节点（旁路 wrap hook 保证，见 docs/superpowers/specs/plugin.md 第 15 节）；`invoke(input, config, *, context, **kwargs)` / `stream(...)` 与 `graph.ainvoke/astream` **完全同形透传**（stream 是普通函数，产出原始 `(mode, payload)` 元组，不做事件分类）；**context 必传**（core 运行时断言，langgraph 对 None 静默放行）；`loop.graph` 暴露 `CompiledStateGraph`（agent 层 repair/compress 用 `aget_state/aupdate_state`）。默认节点在构造时捕获 config 截断参数（构造后改配置不生效）
```

约 87 行，「Plugin 模式」段开头 `core/plugin/` 改为 `lang_agent/plugin/`；约 96 行 `tests/test_core/test_plugin` 改为 `tests/test_plugin`。

- [ ] **Step 9: 运行完整门禁**

```bash
.venv/bin/ruff check lang_agent/ tests/
.venv/bin/mypy lang_agent tests --explicit-package-bases
.venv-check/bin/mypy --python-executable .venv/bin/python lang_agent tests --explicit-package-bases
.venv/bin/python -m pytest -q
```

Expected：ruff 通过、两个 mypy 均 `Success: no issues found in 52 source files`、pytest 全绿（总数 159 − 2 删除 + 1 新增 = 158 passed）。

- [ ] **Step 10: 提交**

```bash
git add lang_agent/core/loop/react_agent.py tests/test_core/test_loop/test_react_agent.py AGENTS.md
git commit -m "refactor: AgentLoop 移除自定义节点注入与 wrap 能力声明（plugin.md §15）"
```
