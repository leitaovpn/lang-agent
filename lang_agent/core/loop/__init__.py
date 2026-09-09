"""core.loop：ReAct 循环与生命周期支撑。

graph 构建/编译/同形透传（react_agent）+ 事件协议（events）
+ 历史修复（repair）+ 调用重试（retry）+ context 压缩（compress）；
未来新循环形态（如 plan-execute）在此包新增模块。
"""
from .approval import (
    ToolApprovalDecision,
    ToolApprovalHook,
    ToolApprovalRequest,
    require_human_approval,
)
from .events import AgentEvent, ConversationResult
from .react_agent import (
    AgentContext,
    AgentLoop,
    AgentLoopConfig,
    ReActNode,
    build_checkpointer,
    build_default_agent_node,
    build_default_approval_node,
    build_default_tools_node,
)

__all__ = [
    "AgentContext",
    "AgentEvent",
    "AgentLoop",
    "AgentLoopConfig",
    "ConversationResult",
    "ReActNode",
    "ToolApprovalDecision",
    "ToolApprovalHook",
    "ToolApprovalRequest",
    "build_checkpointer",
    "build_default_agent_node",
    "build_default_approval_node",
    "build_default_tools_node",
    "require_human_approval",
]
