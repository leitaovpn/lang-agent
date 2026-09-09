"""agent.server：FastAPI 对外接口（uvicorn "lang_agent.agent.server:app"）。"""
from .app import app

__all__ = ["app"]
