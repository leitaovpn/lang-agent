# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目指导

本仓库的完整开发指导（命令、架构、约定、已知坑）见 [AGENTS.md](AGENTS.md)——Claude Code 会自动读取该文件，本文件不重复其内容，只补充 Claude Code 特化要点。

## Claude Code 特化要点

- **所有 Python 命令用 `.venv/bin/python` / `.venv/bin/pip`**，venv 是 Python 3.9.6，与系统 python3（3.13）不一致
- 改 `core/` 前先读 AGENTS.md「已知坑」：节点内 LLM 调用必须传 config、State TypedDict 不能加 future annotations、sqlite checkpointer 必须经 `await build_checkpointer(config)` 异步创建
- 测试 LLM 一律用 `tests/conftest.py` 的 `FakeChatModel`，不依赖真实 API key；端到端验证可用临时 OpenAI 兼容 mock（历史做法：/tmp 下的本地 mock 服务 + `DEEPSEEK_BASE_URL` 指向它）
- 该仓库维护有持久化记忆（本会话目录），记录 langgraph 0.6.11 组合的具体坑与解法，做相关改动前可先回忆
