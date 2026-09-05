"""agent 层配置：从环境变量 / .env 读取。"""
import os

from dotenv import load_dotenv

from lang_agent.core import AgentLoopConfig

load_dotenv()

DEFAULT_MODEL = os.environ.get("LANG_AGENT_MODEL", "deepseek-v4-flash")
DEFAULT_PROVIDER = os.environ.get("LANG_AGENT_PROVIDER", "deepseek")
DEFAULT_PROTOCOL = os.environ.get("LANG_AGENT_PROTOCOL", "chat_response")

DEFAULT_HOST = os.environ.get("LANG_AGENT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("LANG_AGENT_PORT", "8000"))


def loop_config() -> AgentLoopConfig:
    """按环境变量装配 AgentLoopConfig（默认 sqlite 持久化多轮记忆）。"""
    return AgentLoopConfig(
        checkpointer_kind=os.environ.get("LANG_AGENT_CHECKPOINTER", "sqlite"),
        db_path=os.path.expanduser(
            os.environ.get("LANG_AGENT_DB_PATH", "~/.lang-agent/checkpoints.sqlite")
        ),
    )
