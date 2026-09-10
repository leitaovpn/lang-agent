"""可选的批量工具审批插件；暂停由固定 before_tool 节点处理。"""

import hashlib
import json
from typing import override

from langchain_core.messages import AIMessage, ToolMessage

from .base import PluginBase
from .types import HookPause, HookResult, PluginError


def fingerprint(call):
    return hashlib.sha256(
        json.dumps(
            {"name": call["name"], "args": call["args"]}, sort_keys=True
        ).encode()
    ).hexdigest()


def parse_answer(answer):
    if isinstance(answer, bool):
        return ("approve" if answer else "reject"), None
    if isinstance(answer, dict):
        action = answer.get("action")
        if action == "edit":
            if not isinstance(answer.get("args"), dict):
                raise PluginError("审批答案编辑必须提供 args 对象")
            return "approve", answer["args"]
    else:
        action = answer
    if action not in ("approve", "reject"):
        raise PluginError("审批答案必须为 approve/reject 或 edit 对象")
    return action, None


class ToolApprovalPlugin(PluginBase):
    name = "tool_approval"

    @override
    def validate_answer(self, key, payload, answer):
        super().validate_answer(key, payload, answer)
        parse_answer(answer)

    @override
    def before_tool(self, state, runtime):
        message = next(
            m for m in reversed(state["messages"]) if isinstance(m, AIMessage)
        )
        decisions = {}
        for call in message.tool_calls:
            key = call["id"]
            if key not in runtime.answers:
                return HookResult(
                    pause=HookPause(
                        key=key,
                        payload={
                            "tool_call": call,
                            "options": ["approve", "reject"],
                            "fingerprint": fingerprint(call),
                        },
                    )
                )
            action, args = parse_answer(runtime.answers[key])
            if args is not None:
                call["args"] = args
            decisions[key] = {"action": action, "fingerprint": fingerprint(call)}
        return HookResult(
            update={
                "messages": [message],
                "plugin_state": {
                    self.name: {"run_id": runtime.run_id, "decisions": decisions}
                },
            }
        )


def enforce_approval(
    state, request, *, plugin_names: tuple[str, ...], executing: bool = False
):
    """在注册插件中查找本轮的审批决定并强制执行。

    决定存放在插件自己的 namespace（按注册名索引）；同一轮只允许
    一个审批来源，多个来源属于配置错误。
    """
    approval = None
    for name in plugin_names:
        candidate = state.get("plugin_state", {}).get(name)
        if (
            isinstance(candidate, dict)
            and candidate.get("run_id") == state.get("run_id")
            and "decisions" in candidate
        ):
            if approval is not None:
                raise PluginError("同一轮存在多个审批决定来源")
            approval = candidate
    if not approval or not state.get("run_id"):
        return None
    decision = approval.get("decisions", {}).get(request.tool_call["id"])
    if decision is None:
        raise PluginError("当前工具没有审批决定")
    if decision["action"] == "reject":
        if executing:
            raise PluginError("工具调用已被拒绝")
        return ToolMessage(
            content="用户拒绝执行此工具",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )
    if fingerprint(request.tool_call) != decision["fingerprint"]:
        raise PluginError("工具参数在审批后发生变化，需要重新审批")
    return None
