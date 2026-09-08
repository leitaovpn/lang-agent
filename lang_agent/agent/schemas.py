"""agent 层请求/响应模型。"""
from typing import Any

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
