"""ai 层基础类型。"""
from dataclasses import dataclass
from typing import Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(frozen=True)
class ProviderConfig:
    """一个 provider 的接入配置。"""

    name: str
    base_url: str
    api_key_env: str
    base_url_env: Optional[str] = None


# 构建函数：接收 (model, api_key, base_url) 等参数，返回一个
# 可 bind_tools / invoke / astream 的 langchain chat model
LLMFactory = Callable[..., BaseChatModel]
