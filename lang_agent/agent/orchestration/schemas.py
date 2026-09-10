"""agent 层请求/响应模型。"""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    model: str = Field(default="deepseek-v4-flash")
    provider: str = Field(default="deepseek")
    protocol: str = Field(default="chat_response")
    thread_id: str = Field(default="default")
    agent_id: str | None = None
    message: str
    system: str | None = None


class ChatResponse(BaseModel):
    agent_id: str = ""
    status: str = "completed"
    interrupts: list[dict[str, Any]] = Field(default_factory=list)
    thread_id: str
    answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ResumeRequest(ChatRequest):
    """只接受服务器已知 agent 上的待审批 id，不追加用户消息。"""

    message: str = ""
    agent_id: str
    answers: dict[str, Any] | None = None
