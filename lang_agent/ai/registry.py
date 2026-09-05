"""provider/protocol 注册表：把 (model, provider, protocol) 映射为 LLM 调用对象。"""
import os
from typing import Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from lang_agent.ai.base import LLMFactory, ProviderConfig
from lang_agent.ai.errors import (
    MissingApiKeyError,
    UnknownProviderError,
    UnknownProtocolError,
)


class ProviderRegistry:
    """按 provider + protocol 两级分发 LLM 构建工厂。"""

    def __init__(self):
        self._factories: Dict[str, Dict[str, LLMFactory]] = {}
        self._configs: Dict[str, ProviderConfig] = {}

    def register(
        self,
        provider: str,
        protocols: Dict[str, LLMFactory],
        config: ProviderConfig,
    ) -> None:
        """注册一个 provider 及其支持的 protocol 集合。"""
        self._configs[provider] = config
        self._factories[provider] = dict(protocols)

    def build(
        self,
        *,
        model: str,
        provider: str = "deepseek",
        protocol: str = "chat_response",
        api_key: Optional[str] = None,
    ) -> BaseChatModel:
        """根据 (model, provider, protocol) 返回 LLM 调用对象。

        异常语义：
        - UnknownProviderError / UnknownProtocolError：调用方参数错误
        - MissingApiKeyError：配置缺失（缺环境变量）
        """
        config = self._configs.get(provider)
        if config is None:
            raise UnknownProviderError(
                "未知 provider: %r，已注册: %s" % (provider, sorted(self._factories))
            )

        factory = self._factories[provider].get(protocol)
        if factory is None:
            raise UnknownProtocolError(
                "provider %r 不支持 protocol %r，支持: %s"
                % (provider, protocol, sorted(self._factories[provider]))
            )

        if api_key is None:
            api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise MissingApiKeyError(
                "缺少 API key：请设置环境变量 %s 或显式传入 api_key 参数"
                % config.api_key_env
            )

        base_url = config.base_url
        if config.base_url_env:
            base_url = os.environ.get(config.base_url_env, base_url)

        return factory(model=model, api_key=api_key, base_url=base_url)


# 全局单例注册表；各 provider 模块 import 时自行 register
registry = ProviderRegistry()


def get_llm(
    *,
    model: str,
    provider: str = "deepseek",
    protocol: str = "chat_response",
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """便捷工厂。示例：

    get_llm(model="deepseek-v4-flash", provider="deepseek", protocol="chat_response")
    """
    return registry.build(model=model, provider=provider, protocol=protocol, api_key=api_key)
