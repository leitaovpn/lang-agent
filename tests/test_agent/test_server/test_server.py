"""agent 层 FastAPI 接口测试：用 FakeChatModel 注入，不依赖真实 API。"""
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from lang_agent.agent.orchestration import ChatSession
from lang_agent.agent.orchestration.deps import ChatDeps
from lang_agent.agent.server.app import app, build_deps
from lang_agent.core.loop import AgentContext, AgentLoop, AgentLoopConfig
from lang_agent.core.tool import instantiate_tools
from tests.conftest import FakeChatModel

TOOL_CALL = {
    "name": "calculator",
    "args": {"expression": "(3+5)*7"},
    "id": "call_1",
    "type": "tool_call",
}


@pytest.fixture
def override_deps():
    """把 build_deps 依赖替换为给定脚本的 FakeChatModel 会话；返回清理函数。"""

    def set_override(script):
        async def _factory(request=None):
            llm = FakeChatModel(responses=list(script))
            loop = AgentLoop(config=AgentLoopConfig(), checkpointer=InMemorySaver())
            return ChatDeps(
                session=ChatSession(loop=loop, config=AgentLoopConfig()),
                context=AgentContext(llm=llm, tools=instantiate_tools()),
            )

        app.dependency_overrides[build_deps] = _factory
        return lambda: app.dependency_overrides.clear()

    yield set_override
    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_chat_endpoint(client, override_deps):
    cleanup = override_deps([AIMessage(content="答案是 42")])
    try:
        resp = await client.post("/chat", json={"message": "1+1 等于几"})
    finally:
        cleanup()
    assert resp.status_code == 200
    data = resp.json()
    assert data["thread_id"] == "default"
    assert data["answer"] == "答案是 42"
    assert data["tool_calls"] == []


async def test_chat_endpoint_with_tool_round(client, override_deps):
    cleanup = override_deps(
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
    # 必须 importlib 拿真模块:写成 `import lang_agent.agent.server.app as server_mod`
    # 会经父包属性链绑定 server/__init__ 重导出的 FastAPI 实例(同名 app 属性
    # 遮蔽子模块),补丁打到实例上静默失效——build_deps 读的是模块全局 get_deps。
    import importlib

    server_mod = importlib.import_module("lang_agent.agent.server.app")

    async def boom(request=None, **kwargs):
        from lang_agent.ai.errors import UnknownProviderError

        raise UnknownProviderError("monkeypatch-marker")

    monkeypatch.setattr(server_mod, "get_deps", boom)
    resp = await client.post("/chat", json={"message": "hi", "provider": "nope"})
    assert resp.status_code == 400
    assert "monkeypatch-marker" in resp.json()["detail"]


async def test_stream_endpoint_sse(client, override_deps):
    cleanup = override_deps(
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


async def test_http_approval_resume(client):
    from lang_agent.plugin import PluginRegistry
    from lang_agent.plugin.approval import ToolApprovalPlugin
    loop = AgentLoop()
    registry = PluginRegistry()
    registry.register(ToolApprovalPlugin(), agent_id=loop.agent_id, hooks=['before_tool'])
    loop.update_plugin_hooks(registry.snapshot(agent_id=loop.agent_id))
    deps = ChatDeps(session=ChatSession(loop=loop), context=AgentContext(llm=FakeChatModel(responses=[AIMessage(content='', tool_calls=[TOOL_CALL]), AIMessage(content='完成')]), tools=instantiate_tools()))
    from lang_agent.agent.server.app import build_resume_deps
    app.dependency_overrides[build_resume_deps] = lambda: deps
    app.dependency_overrides[build_deps] = lambda: deps
    try:
        response = await client.post('/chat', json={'message': '算一下'})
        data = response.json()
        assert data['status'] == 'interrupted'
        response = await client.post('/chat/resume', json={'agent_id': loop.agent_id, 'answers': {data['interrupts'][0]['id']: 'approve'}})
        assert response.status_code == 200
        assert response.json()['answer'] == '完成'
    finally:
        app.dependency_overrides.clear()
