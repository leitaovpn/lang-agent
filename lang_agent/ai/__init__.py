"""ai 层：通过 (model, provider, protocol) 获取对应的 LLM 调用对象。"""
from lang_agent.ai import chat  # noqa: F401  触发 provider 注册
from lang_agent.ai.registry import get_llm, registry

__all__ = ["get_llm", "registry"]
