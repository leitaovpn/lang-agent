"""agent 层装配：把 ai 层 + core 层组合成可用的会话依赖（按参数缓存单例）。"""

import asyncio
import json
from dataclasses import dataclass
from uuid import uuid4

from lang_agent.ai import get_llm
from lang_agent.core.loop import (
    AgentContext,
    AgentLoop,
    AgentLoopConfig,
    build_checkpointer,
)
from lang_agent.core.tool import instantiate_tools

from . import config as app_config
from .session import ChatSession


@dataclass(slots=True)
class ChatDeps:
    """一次请求的会话依赖：session（graph/checkpointer 单例）+ context（llm/tools 单例）。"""

    session: ChatSession
    context: AgentContext


# (model, provider, protocol, checkpointer_kind, db_path) → ChatDeps 单例
_deps_lock = asyncio.Lock()
_deps: dict[tuple[str, str, str, str, str], ChatDeps] = {}


async def get_deps(
    *,
    model: str | None = None,
    provider: str | None = None,
    protocol: str | None = None,
    config: AgentLoopConfig | None = None,
    agent_id: str | None = None,
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
    async with _deps_lock:
        if agent_id:
            existing = next(
                (d for d in _deps.values() if d.session.loop.agent_id == agent_id), None
            )
            if existing is not None:
                return existing
        if key not in _deps:
            llm = get_llm(model=model, provider=provider, protocol=protocol)
            checkpointer = await build_checkpointer(cfg)
            restored_id = None
            if cfg.checkpointer_kind == "sqlite":
                conn = checkpointer.conn
                await conn.execute(
                    "CREATE TABLE IF NOT EXISTS lang_agent_identity (identity_key TEXT PRIMARY KEY, agent_id TEXT NOT NULL UNIQUE)"
                )
                identity_key = json.dumps([model, provider, protocol])
                await conn.execute(
                    "INSERT OR IGNORE INTO lang_agent_identity VALUES (?, ?)",
                    (identity_key, uuid4().hex),
                )
                await conn.commit()
                async with conn.execute(
                    "SELECT agent_id FROM lang_agent_identity WHERE identity_key = ?",
                    (identity_key,),
                ) as cursor:
                    row = await cursor.fetchone()
                    restored_id = row[0]
            if agent_id and agent_id != restored_id:
                if cfg.checkpointer_kind == "sqlite":
                    await checkpointer.conn.close()
                raise ValueError("未知 agent_id，不能通过请求创建 agent")
            loop = (
                AgentLoop.restore(
                    agent_id=restored_id, config=cfg, checkpointer=checkpointer
                )
                if restored_id
                else AgentLoop(config=cfg, checkpointer=checkpointer)
            )
            _deps[key] = ChatDeps(
                session=ChatSession(loop=loop, config=cfg),
                context=AgentContext(llm=llm, tools=instantiate_tools()),
            )
        result = _deps[key]
        if agent_id and result.session.loop.agent_id != agent_id:
            raise ValueError("agent_id 与装配配置不匹配")
        return result
