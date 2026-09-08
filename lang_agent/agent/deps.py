"""agent 层装配：把 ai 层 + core 层组合成可用的 AgentLoop（按参数缓存单例）。"""
from lang_agent.ai import get_llm
from lang_agent.agent import config as app_config
from lang_agent.core import AgentLoop, AgentLoopConfig, build_checkpointer

# (model, provider, protocol, checkpointer_kind, db_path) → AgentLoop 单例
_loops: dict[tuple[str, str, str, str, str], AgentLoop] = {}


async def get_loop(
    *,
    model: str | None = None,
    provider: str | None = None,
    protocol: str | None = None,
    config: AgentLoopConfig | None = None,
) -> AgentLoop:
    """按参数装配 AgentLoop；相同参数返回同一个实例（同一 checkpointer 连接）。

    ai 层错误（未知 provider/protocol、缺 API key）原样向上抛出，
    由 server 层映射为 HTTP 错误。
    """
    model = model or app_config.DEFAULT_MODEL
    provider = provider or app_config.DEFAULT_PROVIDER
    protocol = protocol or app_config.DEFAULT_PROTOCOL
    cfg = config or app_config.loop_config()

    key = (model, provider, protocol, cfg.checkpointer_kind, cfg.db_path)
    if key not in _loops:
        llm = get_llm(model=model, provider=provider, protocol=protocol)
        checkpointer = await build_checkpointer(cfg)
        _loops[key] = AgentLoop(llm, config=cfg, checkpointer=checkpointer)
    return _loops[key]
