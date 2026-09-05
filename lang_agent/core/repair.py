"""checkpoint 里的历史修复：保证 tool_call 不变式。

LLM 对消息历史的硬性要求（本项目必须满足）：
1. AIMessage 若含 tool_calls，其后必须为每条 tool_call 配一条对应的 ToolMessage；
2. tool_call_id 不重合——去重范围只限「两个 AIMessage 之间」，之前的
   tool_call_id 不参与处理。

使用位置：invoke/stream 入口（_repair_checkpoint_state）——上一轮产生的坏段
（重复 id、缺失 ToolMessage 等）在 checkpoint 里按原样存储，本次调用开始前
修复并写回，避免下次 checkpoint 拉取到错误消息。写回用 RemoveMessage 全删
再加回（add_messages reducer 会把新消息追加到末尾，直接返回修复列表会打乱顺序）。

修复范围：最后一条**带 tool_calls/invalid_tool_calls** 的 AIMessage 起的段——
坏段可能被本轮末尾的纯文本 AIMessage 推到「最后一条 AIMessage」之前。

修复策略：
- 段内 tool_call_id 缺失/重复 → 改名去重（前缀 id 不参与）；
- 缺失的 ToolMessage → 合成「工具调用未返回结果」错误消息补齐；
- 重复/孤儿 ToolMessage → 丢弃；
- invalid_tool_calls（解析失败的调用）→ 合成「格式错误」反馈 ToolMessage，
  其 id 形如 invalid_<aimessage下标>_<条内序号>，幂等（已存在则不重复合成）。
"""
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

INVALID_ID_PREFIX = "invalid_"


def _unique_id(used: set, base: str) -> str:
    call_id = base
    k = 1
    while call_id in used:
        call_id = f"{base}_dup{k}"
        k += 1
    used.add(call_id)
    return call_id


def _last_calling_aimessage_index(messages: list[BaseMessage]) -> int:
    """最后一条带 tool_calls / invalid_tool_calls 的 AIMessage 下标。"""
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls):
            return idx
    return -1


def _repair_from(messages: list[BaseMessage], start_ai: int) -> list[BaseMessage]:
    """从下标 start_ai 的 AIMessage 起做尾部修复（start_ai 之后的消息保留）。"""
    # 前缀按归纳不变式信任；其 tool_call_id 不参与去重
    # （唯一性只要求「两个 AIMessage 之间」不重合）
    repaired = list(messages[:start_ai])
    used: set = set()

    message = cast(AIMessage, messages[start_ai])
    # 1) 尾部 tool_calls id 唯一化（与历史全局去重）
    calls = []
    for k, call in enumerate(message.tool_calls or []):
        call = dict(call)
        call_id_base = call.get("id")
        call["id"] = _unique_id(
            used, str(call_id_base) if call_id_base else f"call_{start_ai}_{k}"
        )
        calls.append(call)
    # 无改写的消息原样保留（避免 tool_calls=[] 被改成 None 破坏相等性）
    ai = message.model_copy(update={"tool_calls": calls}) if calls else message
    repaired.append(ai)

    # 2) 收集紧随其后的 ToolMessage
    j = start_ai + 1
    tool_msgs: list[ToolMessage] = []
    while j < len(messages) and isinstance(messages[j], ToolMessage):
        tool_msgs.append(cast(ToolMessage, messages[j]))
        j += 1

    call_ids = {c["id"] for c in calls}
    if calls and len(tool_msgs) == len(calls):
        # 数量一致：按位置一一对应（兼容原始 id 重复的情况）
        for call, tm in zip(calls, tool_msgs):
            repaired.append(tm.model_copy(update={"tool_call_id": call["id"]}))
    else:
        # 数量不一致：按 id 匹配；重复/孤儿丢弃，缺失的补错误消息
        seen = set()
        for tm in tool_msgs:
            tid = tm.tool_call_id
            if tid in seen:
                continue
            if tid in call_ids or tid.startswith(INVALID_ID_PREFIX):
                seen.add(tid)
                repaired.append(tm)
        for call in calls:
            if call["id"] not in seen:
                repaired.append(
                    ToolMessage(
                        content="工具调用未返回结果",
                        tool_call_id=call["id"],
                        name=call.get("name") or "unknown_tool",
                    )
                )

    # 3) invalid_tool_calls → 错误反馈（确定性 id，幂等）
    following_ids = {tm.tool_call_id for tm in tool_msgs}
    for k, invalid in enumerate(message.invalid_tool_calls or []):
        tid = f"{INVALID_ID_PREFIX}{start_ai}_{k}"
        if tid in following_ids:
            continue
        repaired.append(
            ToolMessage(
                content="工具调用格式错误: %s"
                % (invalid.get("error") or "参数解析失败"),
                tool_call_id=tid,
                name=invalid.get("name") or "unknown_tool",
            )
        )

    # 尾部之后的非 tool 消息原样保留
    repaired.extend(messages[j:])
    return repaired


def repair_state_for_checkpoint(messages: list[BaseMessage]) -> list[BaseMessage]:
    """checkpoint 持久化修复：从最后一条带 tool_calls/invalid_tool_calls 的
    AIMessage 起修复（原列表不被修改）。

    坏段可能被本轮最终的纯文本 AIMessage 推到「最后一条 AIMessage」之前，
    因此修复要越过它，修到最后一个「调用工具」的段。
    """
    start_ai = _last_calling_aimessage_index(messages)
    if start_ai < 0:
        return list(messages)
    return _repair_from(messages, start_ai)
