"""ai 层注册表测试：get_llm(model, provider, protocol) 的分发与错误语义。"""
import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from lang_agent.ai import get_llm
from lang_agent.ai.errors import (
    MissingApiKeyError,
    UnknownProtocolError,
    UnknownProviderError,
)


def test_build_deepseek_chat_response(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    llm = get_llm(model="deepseek-v4-flash", provider="deepseek", protocol="chat_response")
    assert isinstance(llm, BaseChatModel)
    assert llm.model_name == "deepseek-v4-flash"
    assert llm.openai_api_base.rstrip("/") == "https://api.deepseek.com"


def test_build_uses_explicit_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    llm = get_llm(model="deepseek-v4-flash", api_key="sk-explicit")
    assert llm.openai_api_key.get_secret_value() == "sk-explicit"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError) as exc_info:
        get_llm(model="deepseek-v4-flash")
    assert "DEEPSEEK_API_KEY" in str(exc_info.value)


def test_unknown_provider_raises():
    with pytest.raises(UnknownProviderError) as exc_info:
        get_llm(model="x", provider="nope")
    assert "nope" in str(exc_info.value)


def test_unknown_protocol_raises(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with pytest.raises(UnknownProtocolError) as exc_info:
        get_llm(model="x", provider="deepseek", protocol="nope")
    assert "nope" in str(exc_info.value)


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://custom.example.com/v1")
    llm = get_llm(model="deepseek-v4-flash")
    assert llm.openai_api_base.rstrip("/") == "https://custom.example.com/v1"
