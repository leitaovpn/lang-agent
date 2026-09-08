"""deepseek provider 注册：OpenAI 兼容 chat 接口（protocol=chat_response）。"""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from lang_agent.ai.base import ProviderConfig
from lang_agent.ai.registry import registry


def _chat_response_factory(
    *, model: str, api_key: str, base_url: str, **kwargs: Any
) -> BaseChatModel:
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, **kwargs)


registry.register(
    provider="deepseek",
    protocols={"chat_response": _chat_response_factory},
    config=ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
    ),
)
