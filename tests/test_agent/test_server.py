"""agent 层 FastAPI 接口测试：用 FakeChatModel 注入，不依赖真实 API。"""
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from lang_agent.agent.server import app, build_loop
from lang_agent.core import AgentLoop
from tests.conftest import FakeChatModel

TOOL_CALL = {
    "name": "calculator",
    "args": {"expression": "(3+5)*7"},
    "id": "call_1",
    "type": "tool_call",
}


@pytest.fixture
def override_loop():
    """把 build_loop 依赖替换为给定脚本的 FakeChatModel 循环；返回清理函数。"""

    def set_override(script):
        async def _factory(request=None):
            return AgentLoop(llm=FakeChatModel(responses=list(script)))

        app.dependency_overrides[build_loop] = _factory
        return lambda: app.dependency_overrides.clear()

    yield set_override
    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_chat_endpoint(client, override_loop):
    cleanup = override_loop([AIMessage(content="答案是 42")])
    try:
        resp = await client.post("/chat", json={"message": "1+1 等于几"})
    finally:
        cleanup()
    assert resp.status_code == 200
    data = resp.json()
    assert data["thread_id"] == "default"
    assert data["answer"] == "答案是 42"
    assert data["tool_calls"] == []


async def test_chat_endpoint_with_tool_round(client, override_loop):
    cleanup = override_loop(
        [AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="结果是 56")]
    )
    try:
        resp = await client.post("/chat", json={"message": "计算 (3+5)*7"})
    finally:
        cleanup()
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "结果是 56"
    assert [tc["name"] for tc in data["tool_calls"]] == ["calculator"]


async def test_chat_unknown_provider_returns_400(client, monkeypatch):
    import lang_agent.agent.server as server_mod

    async def boom(request=None, **kwargs):
        from lang_agent.ai.errors import UnknownProviderError

        raise UnknownProviderError("未知 provider: 'nope'，已注册: ['deepseek']")

    monkeypatch.setattr(server_mod, "get_loop", boom)
    resp = await client.post("/chat", json={"message": "hi", "provider": "nope"})
    assert resp.status_code == 400
    assert "nope" in resp.json()["detail"]


async def test_stream_endpoint_sse(client, override_loop):
    cleanup = override_loop(
        [AIMessage(content="", tool_calls=[TOOL_CALL]), AIMessage(content="结果是 56")]
    )
    try:
        resp = await client.post("/chat/stream", json={"message": "计算 (3+5)*7"})
    finally:
        cleanup()
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: llm_token" in body
    assert "event: tool_call" in body
    assert "event: tool_result" in body
    assert "event: done" in body
    assert "结果是 56" in body
