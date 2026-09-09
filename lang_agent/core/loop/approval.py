"""工具执行门禁的数据协议与默认人工审批钩子。"""
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal, NotRequired, TypedDict, cast

from langgraph.types import interrupt


class ToolApprovalCall(TypedDict):
    """提交给门禁的一条工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolApprovalRequest(TypedDict):
    """一次审批批次；同一 AIMessage 的工具调用组成一个批次。"""

    approval_id: str
    tool_calls: list[ToolApprovalCall]


class ToolApprovalItem(TypedDict):
    """一条工具调用的人工决定。"""

    tool_call_id: str
    action: Literal["approve", "reject"]
    reason: NotRequired[str]


class ToolApprovalDecision(TypedDict):
    """门禁钩子的返回值。"""

    approval_id: str
    decisions: list[ToolApprovalItem]


type ToolApprovalHook = Callable[
    [ToolApprovalRequest], ToolApprovalDecision | Awaitable[ToolApprovalDecision]
]


def build_approval_request(tool_calls: list[dict[str, Any]]) -> ToolApprovalRequest:
    """从模型工具调用构造稳定、可序列化的审批请求。"""
    calls: list[ToolApprovalCall] = [
        {
            "id": str(call.get("id") or ""),
            "name": str(call.get("name") or ""),
            "arguments": cast(dict[str, Any], call.get("args") or {}),
        }
        for call in tool_calls
    ]
    raw = json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    approval_id = "approval_" + hashlib.sha256(raw.encode()).hexdigest()[:20]
    return {"approval_id": approval_id, "tool_calls": calls}


def require_human_approval(request: ToolApprovalRequest) -> ToolApprovalDecision:
    """默认门禁：暂停 graph，把本批工具调用交给外部人工决定。"""
    return cast(ToolApprovalDecision, interrupt(request))


def approve_all(request: ToolApprovalRequest) -> ToolApprovalDecision:
    """未配置门禁时保持原有行为：自动放行全部工具。"""
    return {
        "approval_id": request["approval_id"],
        "decisions": [
            {"tool_call_id": call["id"], "action": "approve"}
            for call in request["tool_calls"]
        ],
    }


def validate_approval_decision(
    request: ToolApprovalRequest, decision: ToolApprovalDecision
) -> ToolApprovalDecision:
    """拒绝缺项、重复项、过期批次和未知动作，避免绕过门禁。"""
    if decision.get("approval_id") != request["approval_id"]:
        raise ValueError("审批批次不匹配或已过期")
    expected = [call["id"] for call in request["tool_calls"]]
    items = decision.get("decisions")
    if not isinstance(items, list):
        raise TypeError("审批决定缺少 decisions")
    received: list[str] = []
    for item in items:
        tool_call_id = item.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            raise TypeError("审批决定缺少 tool_call_id")
        received.append(tool_call_id)
    if len(received) != len(expected) or sorted(received) != sorted(expected):
        raise ValueError("审批决定必须完整覆盖本批工具调用")
    for item in items:
        if item.get("action") not in ("approve", "reject"):
            raise ValueError("审批动作只能是 approve 或 reject")
    return decision
