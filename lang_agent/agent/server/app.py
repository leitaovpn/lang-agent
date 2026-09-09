"""agent 层：FastAPI 对外接口。

- POST /chat        同步：返回最终回答与工具调用汇总
- POST /chat/stream SSE 流式：llm_token / tool_call / tool_result / done / error
- POST /chat/resume 与 /chat/resume/stream：提交工具审批并恢复原运行
- GET /chat/approval：查询线程待审批批次
"""
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from lang_agent.agent.orchestration.config import (
    DEFAULT_MODEL,
    DEFAULT_PROTOCOL,
    DEFAULT_PROVIDER,
)
from lang_agent.agent.orchestration.deps import ChatDeps, get_deps
from lang_agent.agent.orchestration.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
)
from lang_agent.agent.orchestration.session import (
    ApprovalNotFoundError,
    ApprovalPendingError,
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
    if isinstance(exc, MissingApiKeyError):
        return HTTPException(status_code=500, detail=str(exc))
    if isinstance(exc, ApprovalPendingError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ApprovalNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


async def build_deps(request: ChatRequest) -> ChatDeps:
    """按请求参数装配会话依赖（可被 dependency_overrides 替换以注入测试 fake）。"""
    try:
        return await get_deps(
            model=request.model, provider=request.provider, protocol=request.protocol
        )
    except Exception as exc:  # noqa: BLE001 统一映射 HTTP 错误，不外泄 traceback
        raise _map_error(exc)


async def build_resume_deps(request: ChatResumeRequest) -> ChatDeps:
    """按续跑请求的模型参数取得原线程依赖。"""
    try:
        return await get_deps(
            model=request.model, provider=request.provider, protocol=request.protocol
        )
    except Exception as exc:  # noqa: BLE001
        raise _map_error(exc)


def _chat_response(thread_id: str, result) -> ChatResponse:
    return ChatResponse(
        thread_id=thread_id,
        answer=result.final_text,
        tool_calls=result.tool_calls,
        status=result.status,
        approval=result.approval,
    )


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
    return _chat_response(request.thread_id, result)


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


@app.post("/chat/resume", response_model=ChatResponse)
async def chat_resume(
    request: ChatResumeRequest,
    deps: ChatDeps = Depends(build_resume_deps),  # noqa: B008
):
    decision = {
        "approval_id": request.approval_id,
        "decisions": [item.model_dump(exclude_none=True) for item in request.decisions],
    }
    try:
        result = await deps.session.resume(
            decision, thread_id=request.thread_id, context=deps.context
        )
    except Exception as exc:  # noqa: BLE001
        raise _map_error(exc)
    return _chat_response(request.thread_id, result)


@app.post("/chat/resume/stream")
async def chat_resume_stream(
    request: ChatResumeRequest,
    deps: ChatDeps = Depends(build_resume_deps),  # noqa: B008
):
    decision = {
        "approval_id": request.approval_id,
        "decisions": [item.model_dump(exclude_none=True) for item in request.decisions],
    }

    async def events() -> AsyncIterator[str]:
        async for event in deps.session.resume_stream(
            decision, thread_id=request.thread_id, context=deps.context
        ):
            yield event.to_sse()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chat/approval")
async def chat_pending_approval(
    thread_id: str,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    protocol: str = DEFAULT_PROTOCOL,
):
    """查询线程的待审批批次，支持客户端退出后重新接入。"""
    try:
        deps = await get_deps(model=model, provider=provider, protocol=protocol)
        approval = await deps.session.get_pending_approval(thread_id)
    except Exception as exc:  # noqa: BLE001
        raise _map_error(exc)
    if approval is None:
        raise HTTPException(status_code=404, detail="当前线程没有待审批的工具调用")
    return {"thread_id": thread_id, "approval": approval}
