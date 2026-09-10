"""lang-agent：基于 langgraph 的 lang agent。

分层：agent（FastAPI/CLI）→ core（ReAct agent loop）→ ai（LLM 接入）；
plugin（横向插件协议包，core.loop 与 agent 层共用，不反向导入分层）。
"""
