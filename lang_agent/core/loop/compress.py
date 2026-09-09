"""LLM context 压缩的纯逻辑：切点规则、摘要渲染、工具输出截断。

压缩策略（与 _repair_checkpoint_state 并列的入口步骤执行）：
- 旧轮次（保留窗口之前）渲染成文本，交给摘要模型总结，摘要存 AgentState.summary；
- 切点必须落在「轮起点」（最近一条 HumanMessage）——保证 AIMessage(tool_calls)
  与其 ToolMessage 永不拆散，压缩后历史仍满足 repair 不变式；
- 超长工具输出在发送视图截断（不写回 checkpoint，完整内容可追溯）。
"""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def estimate_tokens(messages: list[BaseMessage], llm=None) -> int:
    """token 估算：优先 llm.get_num_tokens_from_messages（需 tokenizer 依赖，
    缺失时抛 ImportError），失败退回字符/4 近似——确定性兜底，不静默跳过。"""
    if llm is not None:
        try:
            return llm.get_num_tokens_from_messages(messages)
        except Exception:  # noqa: BLE001, S110 兜底近似估算
            pass
    return sum(len(m.content) // 4 + 1 for m in messages if isinstance(m.content, str))


def find_round_start(messages: list[BaseMessage], index: int) -> int:
    """从 index 向前找最近的 HumanMessage 下标（压缩切点必须落在轮起点）。"""
    for i in range(min(index, len(messages) - 1), -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return 0


def render_messages_for_summary(messages: list[BaseMessage]) -> str:
    """把待压缩历史渲染成文本，供摘要模型阅读。"""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"user: {message.content}")
        elif isinstance(message, ToolMessage):
            lines.append(f"tool({message.name}): {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"assistant: {message.content or ''}")
            for call in message.tool_calls or []:
                lines.append(
                    f"assistant 调用工具 {call.get('name')}({call.get('args')})"
                )
        else:
            lines.append(
                f"{type(message).__name__}: {getattr(message, 'content', '')}"
            )
    return "\n".join(lines)


def truncate_tool_outputs(messages: list[BaseMessage], max_chars: int) -> list[BaseMessage]:
    """发送视图截断：超长 ToolMessage content 截断并标注；只缩 content 不动结构。"""
    if max_chars <= 0:
        return list(messages)
    truncated: list[BaseMessage] = []
    for message in messages:
        if (
            isinstance(message, ToolMessage)
            and isinstance(message.content, str)
            and len(message.content) > max_chars
        ):
            truncated.append(
                message.model_copy(
                    update={"content": message.content[:max_chars] + "…(已截断)"}
                )
            )
        else:
            truncated.append(message)
    return truncated
