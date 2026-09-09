"""agent 层请求/响应模型。"""
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    model: str = Field(default="deepseek-v4-flash")
    provider: str = Field(default="deepseek")
    protocol: str = Field(default="chat_response")
    thread_id: str = Field(default="default")
    message: str
    system: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["completed", "awaiting_approval"] = "completed"
    approval: dict[str, Any] | None = None


class ApprovalDecision(BaseModel):
    tool_call_id: str
    action: Literal["approve", "reject"]
    reason: str | None = None


class ChatResumeRequest(BaseModel):
    model: str = Field(default="deepseek-v4-flash")
    provider: str = Field(default="deepseek")
    protocol: str = Field(default="chat_response")
    thread_id: str = Field(default="default")
    approval_id: str
    decisions: list[ApprovalDecision]
