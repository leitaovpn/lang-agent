"""agent 层：FastAPI 对外接口。

- POST /chat        同步：返回最终回答与工具调用汇总
- POST /chat/stream SSE 流式：llm_token / tool_call / tool_result / done / error
"""

from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from lang_agent.agent.orchestration.deps import ChatDeps, get_deps
from lang_agent.agent.orchestration.schemas import (
    ChatRequest,
    ChatResponse,
    ResumeRequest,
)
from lang_agent.ai.errors import (
    MissingApiKeyError,
    UnknownProtocolError,
    UnknownProviderError,
)

app = FastAPI(title="lang-agent")


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (UnknownProviderError, UnknownProtocolError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, MissingApiKeyError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


async def build_deps(request: ChatRequest) -> ChatDeps:
    """按请求参数装配会话依赖（可被 dependency_overrides 替换以注入测试 fake）。"""
    try:
        return await get_deps(
            model=request.model,
            provider=request.provider,
            protocol=request.protocol,
            agent_id=request.agent_id,
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
        agent_id=result.agent_id,
        status=result.status,
        interrupts=result.interrupts,
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


async def build_resume_deps(request: ResumeRequest) -> ChatDeps:
    """恢复只定位已有 agent，不能由请求创建新身份。"""
    return await build_deps(request)


@app.post("/chat/resume", response_model=ChatResponse)
async def resume_chat(
    request: ResumeRequest,
    deps: ChatDeps = Depends(build_resume_deps),  # noqa: B008
):
    if request.agent_id != deps.session.loop.agent_id:
        raise HTTPException(status_code=409, detail="agent_id 不匹配")
    try:
        result = await deps.session.resume(
            thread_id=request.thread_id, answers=request.answers, context=deps.context
        )
    except Exception as exc:  # noqa: BLE001 映射公开错误
        raise _map_error(exc)
    return ChatResponse(
        agent_id=result.agent_id,
        thread_id=result.thread_id,
        answer=result.final_text,
        tool_calls=result.tool_calls,
        status=result.status,
        interrupts=result.interrupts,
    )


@app.post("/chat/resume/stream")
async def resume_chat_stream(
    request: ResumeRequest,
    deps: ChatDeps = Depends(build_resume_deps),  # noqa: B008
):
    if request.agent_id != deps.session.loop.agent_id:
        raise HTTPException(status_code=409, detail="agent_id 不匹配")

    async def events():
        async for event in deps.session.resume_stream(
            thread_id=request.thread_id, answers=request.answers, context=deps.context
        ):
            yield event.to_sse()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
