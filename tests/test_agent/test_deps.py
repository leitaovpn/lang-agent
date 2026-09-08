"""agent 层装配（deps/config）测试。"""
from lang_agent.agent import config as app_config
from lang_agent.agent.deps import get_loop
from lang_agent.ai.errors import UnknownProviderError
from lang_agent.core import AgentLoop, AgentLoopConfig


def test_config_defaults():
    assert app_config.DEFAULT_MODEL == "deepseek-v4-flash"
    assert app_config.DEFAULT_PROVIDER == "deepseek"
    assert app_config.DEFAULT_PROTOCOL == "chat_response"


def test_loop_config_kind():
    cfg = app_config.loop_config()
    assert isinstance(cfg, AgentLoopConfig)


def test_loop_config_injects_openai_transient_exceptions():
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    cfg = app_config.loop_config()
    injected = set(cfg.retryable_exceptions or ())
    assert {APIConnectionError, APITimeoutError, InternalServerError, RateLimitError} <= injected


def test_loop_config_respects_env(monkeypatch):
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    cfg = app_config.loop_config()
    assert cfg.checkpointer_kind == "memory"


async def test_get_loop_returns_agent_loop(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    loop = await get_loop(model="deepseek-v4-flash")
    assert isinstance(loop, AgentLoop)


async def test_get_loop_caches_same_params(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    loop1 = await get_loop(model="deepseek-v4-flash")
    loop2 = await get_loop(model="deepseek-v4-flash")
    assert loop1 is loop2


async def test_get_loop_unknown_provider_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LANG_AGENT_CHECKPOINTER", "memory")
    try:
        await get_loop(model="x", provider="nope")
    except UnknownProviderError:
        pass
    else:
        raise AssertionError("应抛出 UnknownProviderError")
