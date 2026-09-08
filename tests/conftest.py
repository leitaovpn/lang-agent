"""共享测试设施：可编程 FakeChatModel。"""
import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field


class FakeChatModel(BaseChatModel):
    """可编程 fake：responses 脚本按调用顺序弹出一条 AIMessage 作为回复。

    - 纯文本 AIMessage → 模拟普通回答（_stream 逐字产出，触发 token 级流式事件）
    - 带 tool_calls 的 AIMessage → 模拟请求调用工具（产出携带 tool_call_chunks 的 chunk）

    每次 _generate/_stream 被调用时，把收到的消息列表记入 seen_messages，
    供测试断言「第 N 轮 LLM 看到了什么」。
    """

    responses: list[AIMessage] = Field(default_factory=list)
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    @override
    def _llm_type(self) -> str:
        return "fake-chat"

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ):
        """记录被绑定的工具，并像真实模型一样返回 RunnableBinding（BaseChatModel.bind_tools 是抽象方法）。"""
        self.bound_tools.extend(tools)
        return self.bind(tools=list(tools))

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeChatModel responses 脚本耗尽")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = cast(
            AIMessage,
            self._generate(messages, stop=stop, run_manager=run_manager, **kwargs).generations[0].message,
        )
        if isinstance(message.content, str):
            for token in message.content:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        if message.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": tc["name"],
                            "args": json.dumps(tc.get("args", {}), ensure_ascii=False),
                            "id": tc["id"],
                            "index": i,
                        }
                        for i, tc in enumerate(message.tool_calls)
                    ],
                )
            )
        if message.invalid_tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    invalid_tool_calls=list(message.invalid_tool_calls),
                )
            )


class FlakyChatModel(FakeChatModel):
    """可配置故障 fake：fail_times 前 N 次调用抛 error，fail_at 指定第 N 次调用抛 error，
    其余按脚本应答。用于重试机制测试；调用次数记入 calls。
    """

    fail_times: int = 1
    fail_at: int | None = None  # 指定第 N 次（1 起）调用失败，用于「中途失败」场景
    error: Any = None
    error_factory: Any = None
    calls: int = 0

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.calls += 1
        should_fail = self.calls <= self.fail_times or self.calls == self.fail_at
        if should_fail:
            error = self.error_factory() if self.error_factory else (self.error or RuntimeError("模型挂了"))
            raise error
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
