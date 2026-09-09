"""graph 调用异常重试：瞬时异常白名单与指数退避。

分层约束：core 不依赖 ai，这里只放框架级传输异常与项目标记异常；
openai 等具体 SDK 的瞬时异常类型由 agent 层注入（AgentLoopConfig.retryable_exceptions）。
"""
import httpx


class RetryableError(Exception):
    """项目标记：任何层 raise 该异常即触发 graph 调用重试。"""


# 默认白名单：网络/超时类 + 项目标记
DEFAULT_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.TransportError,
    TimeoutError,
    ConnectionError,
    RetryableError,
)


def is_retryable(
    exc: BaseException,
    retryable_exceptions: tuple[type[BaseException], ...] | None = None,
) -> bool:
    """白名单判定；retryable_exceptions 为 None 时用默认白名单。"""
    exceptions = retryable_exceptions if retryable_exceptions is not None else DEFAULT_RETRYABLE_EXCEPTIONS
    return isinstance(exc, exceptions)


def compute_delay(attempt: int, base: float = 0.5, factor: float = 2.0) -> float:
    """第 attempt 次失败后的退避秒数（指数）：base, base*factor, base*factor² ..."""
    return base * (factor ** (attempt - 1))
