"""agent 层配置：从环境变量 / .env 读取。"""
import os
from pathlib import Path

from dotenv import load_dotenv

from lang_agent.core.loop import AgentLoopConfig, require_human_approval

load_dotenv()

DEFAULT_MODEL = os.getenv("LANG_AGENT_MODEL", "deepseek-v4-flash")
DEFAULT_PROVIDER = os.getenv("LANG_AGENT_PROVIDER", "deepseek")
DEFAULT_PROTOCOL = os.getenv("LANG_AGENT_PROTOCOL", "chat_response")

DEFAULT_HOST = os.getenv("LANG_AGENT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("LANG_AGENT_PORT", "8000"))


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    """agent 层重试白名单：openai SDK 的瞬时异常（core 层不依赖 ai，无法内置）。"""
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    return (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


def loop_config() -> AgentLoopConfig:
    """按环境变量装配 AgentLoopConfig（默认 sqlite 持久化多轮记忆）。"""
    return AgentLoopConfig(
        checkpointer_kind=os.getenv("LANG_AGENT_CHECKPOINTER", "sqlite"),
        db_path=str(
            Path(
                os.getenv("LANG_AGENT_DB_PATH", "~/.lang-agent/checkpoints.sqlite")
            ).expanduser()
        ),
        retry_max_attempts=int(os.getenv("LANG_AGENT_RETRY_MAX", "3")),
        retry_base_delay=float(os.getenv("LANG_AGENT_RETRY_BASE_DELAY", "0.5")),
        retryable_exceptions=_retryable_exceptions(),
        tool_approval_hook=require_human_approval,
    )
