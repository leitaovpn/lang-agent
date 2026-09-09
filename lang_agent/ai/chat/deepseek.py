"""deepseek provider 注册：OpenAI 兼容 chat 接口（protocol=chat_response）。"""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from lang_agent.ai.base import ProviderConfig
from lang_agent.ai.registry import registry


class DeepSeekChatModel(ChatOpenAI):
    """捞回流式 delta.reasoning_content（thinking）到 chunk.additional_kwargs。

    langchain-openai 1.6 的 _convert_delta_to_message_chunk 会丢弃
    reasoning_content（docstring 明确 not extracted），thinking 只会出现在
    chunk 级流里——这里在 chunk 转换后从原始 delta 取回并附加，供事件层
    classify_message_chunk 分类为 thinking_token。同步/异步流均走此方法。
    """

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> Any:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None:
            return generation
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if choices:
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return generation


def _chat_response_factory(
    *, model: str, api_key: str, base_url: str, **kwargs: Any
) -> BaseChatModel:
    return DeepSeekChatModel(
        model=model, api_key=SecretStr(api_key), base_url=base_url, **kwargs
    )


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
