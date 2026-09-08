"""agent 层装配：把 ai 层 + core 层组合成可用的会话依赖（按参数缓存单例）。"""
from dataclasses import dataclass

from lang_agent.agent import config as app_config
from lang_agent.agent.session import ChatSession
from lang_agent.ai import get_llm
from lang_agent.core import AgentContext, AgentLoop, AgentLoopConfig, build_checkpointer
from lang_agent.core.tool_registry import instantiate_tools


@dataclass(slots=True)
class ChatDeps:
    """一次请求的会话依赖：session（graph/checkpointer 单例）+ context（llm/tools 单例）。"""

    session: ChatSession
    context: AgentContext


# (model, provider, protocol, checkpointer_kind, db_path) → ChatDeps 单例
_deps: dict[tuple[str, str, str, str, str], ChatDeps] = {}


async def get_deps(
    *,
    model: str | None = None,
    provider: str | None = None,
    protocol: str | None = None,
    config: AgentLoopConfig | None = None,
) -> ChatDeps:
    """按参数装配会话依赖；相同参数返回同一个实例（同一 checkpointer 连接与工具集）。

    ai 层错误（未知 provider/protocol、缺 API key）原样向上抛出，
    由 server 层映射为 HTTP 错误（get_llm 抛错发生在写缓存之前，错误参数不会被缓存）。
    """
    model = model or app_config.DEFAULT_MODEL
    provider = provider or app_config.DEFAULT_PROVIDER
    protocol = protocol or app_config.DEFAULT_PROTOCOL
    cfg = config or app_config.loop_config()

    key = (model, provider, protocol, cfg.checkpointer_kind, cfg.db_path)
    if key not in _deps:
        llm = get_llm(model=model, provider=provider, protocol=protocol)
        checkpointer = await build_checkpointer(cfg)
        loop = AgentLoop(config=cfg, checkpointer=checkpointer)
        _deps[key] = ChatDeps(
            session=ChatSession(loop=loop, config=cfg),
            context=AgentContext(llm=llm, tools=instantiate_tools()),
        )
    return _deps[key]
