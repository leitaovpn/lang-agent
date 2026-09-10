"""agent 层装配（deps/config）测试。"""
from lang_agent.agent.orchestration import config as app_config
from lang_agent.agent.orchestration.deps import ChatDeps, get_deps
from lang_agent.ai.errors import UnknownProviderError
from lang_agent.core.loop import AgentLoopConfig


def test_config_defaults():
    assert app_config.DEFAULT_MODEL == "deepseek-v4-flash"
    assert app_config.DEFAULT_PROVIDER == "deepseek"
    assert app_config.DEFAULT_PROTOCOL == "chat_response"


def test_loop_config_kind():
    cfg = app_config.loop_config()
    assert isinstance(cfg, AgentLoopConfig)


def test_loop_config_injects_openai_transient_exceptions():
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    cfg = app_config.loop_config()
    injected = set(cfg.retryable_exceptions or ())
    assert {APIConnectionError, APITimeoutError, InternalServerError, RateLimitError} <= injected


def test_loop_config_respects_env(monkeypatch):
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    cfg = app_config.loop_config()
    assert cfg.checkpointer_kind == "memory"


async def test_get_deps_returns_chat_deps(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    deps = await get_deps(model="deepseek-v4-flash")
    assert isinstance(deps, ChatDeps)


async def test_get_deps_caches_same_params(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    deps1 = await get_deps(model="deepseek-v4-flash")
    deps2 = await get_deps(model="deepseek-v4-flash")
    assert deps1 is deps2


async def test_get_deps_instantiates_tools_once(monkeypatch):
    # 同 key 的 session/context/tools 全部单例：工具链（load_tools 等）只实例化一次
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    deps1 = await get_deps(model="deepseek-v4-flash")
    deps2 = await get_deps(model="deepseek-v4-flash")
    assert deps1.session is deps2.session
    assert deps1.context.tools is deps2.context.tools


async def test_get_deps_unknown_provider_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    try:
        await get_deps(model="x", provider="nope")
    except UnknownProviderError:
        pass
    else:
        raise AssertionError("应抛出 UnknownProviderError")


async def test_sqlite_deps_restores_agent_identity_and_history(tmp_path, monkeypatch):
    import importlib

    from langchain_core.messages import AIMessage

    from tests.conftest import FakeChatModel
    module = importlib.import_module('lang_agent.agent.orchestration.deps')
    monkeypatch.setattr(module, '_deps', {})
    monkeypatch.setattr(module, 'get_llm', lambda **kwargs: FakeChatModel(responses=[AIMessage(content='答')]))
    monkeypatch.setattr(module, 'instantiate_tools', list)
    config = AgentLoopConfig(checkpointer_kind='sqlite', db_path=str(tmp_path / 'identity.sqlite'))
    first = await module.get_deps(config=config)
    agent_id = first.session.loop.agent_id
    await first.session.invoke('上一轮', thread_id='t', context=first.context)
    await first.session.loop.graph.checkpointer.conn.close()
    module._deps.clear()
    second = await module.get_deps(config=config, agent_id=agent_id)
    try:
        assert second.session.loop.agent_id == agent_id
        result = await second.session.invoke('下一轮', thread_id='t', context=second.context)
        assert result.final_text == '答'
        assert second.context.llm.seen_messages[0][0].content == '上一轮'
    finally:
        await second.session.loop.graph.checkpointer.conn.close()
