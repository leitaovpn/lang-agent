"""发给 LLM 的历史修复：保证 tool_call 不变式（只修尾部）。

LLM 对消息历史的硬性要求（本项目必须满足）：
1. AIMessage 若含 tool_calls，其后必须为每条 tool_call 配一条对应的 ToolMessage；
2. tool_call_id 不重合——去重范围只限「两个 AIMessage 之间」（即最后一条
   AIMessage 起的尾部内部），之前的 tool_call_id 不参与处理。

为什么只修尾部：历史只由本循环追加，且每次发给 LLM 的历史都已修复——
LLM 能回复新的 AIMessage，就证明该 AIMessage 之前的结构必然正确（归纳不变式）。
可能违反不变式的只有最后一条 AIMessage 起的尾部：它的 tool_calls 是模型新产出的
（可能解析失败、id 重复/缺失），其后的 ToolMessage 也可能缺失/重复。

修复策略（只作用于发给 LLM 的拷贝，不改 state）：
- 尾部 tool_call_id 缺失/尾部内重复 → 改名去重（前缀 id 不参与）；
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


def repair_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """返回满足 tool_call 不变式的消息列表（原列表不被修改）。"""
    if not messages:
        return []
    last_ai = -1
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], AIMessage):
            last_ai = idx
            break
    if last_ai < 0:
        # 没有 AIMessage（如首轮只有用户消息）：无 tool 结构可修，
        # 只丢弃孤儿 ToolMessage
        return [m for m in messages if not isinstance(m, ToolMessage)]

    # 前缀按归纳不变式信任；其 tool_call_id 不参与去重
    # （唯一性只要求「两个 AIMessage 之间」不重合）
    repaired = list(messages[:last_ai])
    used: set = set()

    message = cast(AIMessage, messages[last_ai])
    # 1) 尾部 tool_calls id 唯一化（与历史全局去重）
    calls = []
    for k, call in enumerate(message.tool_calls or []):
        call = dict(call)
        call_id_base = call.get("id")
        call["id"] = _unique_id(
            used, str(call_id_base) if call_id_base else f"call_{last_ai}_{k}"
        )
        calls.append(call)
    # 无改写的消息原样保留（避免 tool_calls=[] 被改成 None 破坏相等性）
    ai = message.model_copy(update={"tool_calls": calls}) if calls else message
    repaired.append(ai)

    # 2) 收集紧随其后的 ToolMessage
    j = last_ai + 1
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
        tid = f"{INVALID_ID_PREFIX}{last_ai}_{k}"
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
