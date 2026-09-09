"""agent 层请求/响应模型测试。"""
import pytest
from pydantic import ValidationError

from lang_agent.agent.orchestration.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResumeRequest,
)


def test_chat_request_defaults():
    req = ChatRequest(message="你好")
    assert req.model == "deepseek-v4-flash"
    assert req.provider == "deepseek"
    assert req.protocol == "chat_response"
    assert req.thread_id == "default"
    assert req.system is None


def test_chat_request_full():
    req = ChatRequest(
        message="hi",
        model="m",
        provider="p",
        protocol="proto",
        thread_id="t1",
        system="你是助手",
    )
    assert (req.model, req.provider, req.protocol, req.thread_id, req.system) == (
        "m",
        "p",
        "proto",
        "t1",
        "你是助手",
    )


def test_chat_request_requires_message():
    with pytest.raises(ValidationError):
        ChatRequest()


def test_chat_response_default_tool_calls():
    resp = ChatResponse(thread_id="t1", answer="好")
    assert resp.tool_calls == []


def test_chat_resume_request_parses_decisions():
    request = ChatResumeRequest(
        thread_id="t1",
        approval_id="approval_1",
        decisions=[{"tool_call_id": "c1", "action": "approve"}],
    )
    assert request.decisions[0].action == "approve"
