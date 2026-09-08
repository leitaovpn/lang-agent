# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目指导

本仓库的完整开发指导（命令、架构、约定、已知坑）见 [AGENTS.md](AGENTS.md)——Claude Code 会自动读取该文件，本文件不重复其内容，只补充 Claude Code 特化要点。

## Claude Code 特化要点

- **所有 Python 命令用 `.venv_3.13/bin/python` / `.venv_3.13/bin/pip`**，venv 是 Python 3.13.3，与系统 python3 一致（项目 `requires-python = ">=3.13"`）
- 改 `core/` 前先读 AGENTS.md「已知坑」：节点内 LLM 调用必须传 config、State TypedDict 不能加 future annotations、sqlite checkpointer 必须经 `await build_checkpointer(config)` 异步创建、ToolNode 必须显式 `handle_tool_errors=True`、agent_node 合并 chunk 后必须转普通 AIMessage（chunk 合并/序列化会把 invalid_tool_calls 误判为合法调用）
- 测试 LLM 一律用 `tests/conftest.py` 的 `FakeChatModel`，不依赖真实 API key；端到端验证可用临时 OpenAI 兼容 mock（历史做法：/tmp 下的本地 mock 服务 + `DEEPSEEK_BASE_URL` 指向它）
- 该仓库维护有持久化记忆（本会话目录），记录 langgraph 1.2.11 组合的具体坑与解法，做相关改动前可先回忆
