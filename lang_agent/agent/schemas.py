"""agent 层请求/响应模型。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    model: str = Field(default="deepseek-v4-flash")
    provider: str = Field(default="deepseek")
    protocol: str = Field(default="chat_response")
    thread_id: str = Field(default="default")
    message: str
    system: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
