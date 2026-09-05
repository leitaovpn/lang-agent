"""重试纯函数测试：白名单判定与指数退避。"""
import httpx
import pytest

from lang_agent.core.retry import (
    DEFAULT_RETRYABLE_EXCEPTIONS,
    RetryableError,
    compute_delay,
    is_retryable,
)


def test_retryable_error_is_retryable():
    assert is_retryable(RetryableError("限流"))


def test_httpx_transport_errors_are_retryable():
    assert is_retryable(httpx.TimeoutException("超时"))
    assert is_retryable(httpx.TransportError("连接失败"))


def test_builtin_transport_errors_are_retryable():
    assert is_retryable(TimeoutError())
    assert is_retryable(ConnectionError())


def test_deterministic_errors_not_retryable():
    assert not is_retryable(RuntimeError("模型挂了"))
    assert not is_retryable(ValueError("参数错误"))


def test_custom_exception_set_overrides_default():
    class CustomTransient(Exception):
        pass

    assert is_retryable(CustomTransient(), retryable_exceptions=(CustomTransient,))
    assert not is_retryable(RetryableError(), retryable_exceptions=(CustomTransient,))


def test_default_tuple_contains_project_marker():
    assert RetryableError in DEFAULT_RETRYABLE_EXCEPTIONS


def test_compute_delay_exponential():
    assert compute_delay(1, base=0.5, factor=2.0) == 0.5
    assert compute_delay(2, base=0.5, factor=2.0) == 1.0
    assert compute_delay(3, base=0.5, factor=2.0) == 2.0
    assert compute_delay(4, base=1.0, factor=3.0) == 27.0
