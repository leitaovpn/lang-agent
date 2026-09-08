"""agent 层：FastAPI 对外接口。

- POST /chat        同步：返回最终回答与工具调用汇总
- POST /chat/stream SSE 流式：llm_token / tool_call / tool_result / done / error
"""
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from lang_agent.agent.deps import ChatDeps, get_deps
from lang_agent.agent.schemas import ChatRequest, ChatResponse
from lang_agent.ai.errors import (
    MissingApiKeyError,
    UnknownProtocolError,
    UnknownProviderError,
)

app = FastAPI(title="lang-agent")


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (UnknownProviderError, UnknownProtocolError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, MissingApiKeyError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


async def build_deps(request: ChatRequest) -> ChatDeps:
    """按请求参数装配会话依赖（可被 dependency_overrides 替换以注入测试 fake）。"""
    try:
        return await get_deps(
            model=request.model, provider=request.provider, protocol=request.protocol
        )
    except Exception as exc:  # noqa: BLE001 统一映射 HTTP 错误，不外泄 traceback
        raise _map_error(exc)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, deps: ChatDeps = Depends(build_deps)):  # noqa: B008 FastAPI 依赖注入惯用写法
    try:
        result = await deps.session.invoke(
            request.message,
            thread_id=request.thread_id,
            system=request.system,
            context=deps.context,
        )
    except Exception as exc:  # noqa: BLE001 统一映射 HTTP 错误，不外泄 traceback
        raise _map_error(exc)
    return ChatResponse(
        thread_id=request.thread_id,
        answer=result.final_text,
        tool_calls=result.tool_calls,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, deps: ChatDeps = Depends(build_deps)):  # noqa: B008 FastAPI 依赖注入惯用写法
    async def events() -> AsyncIterator[str]:
        async for event in deps.session.stream(
            request.message,
            thread_id=request.thread_id,
            system=request.system,
            context=deps.context,
        ):
            yield event.to_sse()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
