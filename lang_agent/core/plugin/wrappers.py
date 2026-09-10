"""wrap 调用链及同步插件到异步 terminal 的桥接。"""

import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from .registry import Registration
from .types import PluginError

_depth: ContextVar[int] = ContextVar("plugin_wrapper_depth", default=0)


async def call_sync_wrapper(function: Callable, request: Any, handler: Callable) -> Any:
    """同步 handler 在工作线程等待，真实执行仍回到原事件循环。"""
    loop = asyncio.get_running_loop()
    depth = _depth.get()
    if depth >= 8:
        raise PluginError("同步 wrapper 嵌套超过 8 层，请使用异步实现")
    token = _depth.set(depth + 1)
    pending = []

    def execute(req):
        future = asyncio.run_coroutine_threadsafe(handler(req), loop)
        pending.append(future)
        return future.result()

    # 每层独立线程，避免有界共享线程池被嵌套 handler 占满。
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plugin-wrapper")
    context = copy_context()

    # 预绑定参数为无参闭包：run_in_executor 的 TypeVarTuple 签名无法
    # 直接推断 context.run 的 ParamSpec + *args 转发（pyright 报
    # reportArgumentType），闭包同样保留 context 快照语义。
    def run():
        return context.run(function, request, execute)

    try:
        return await loop.run_in_executor(executor, run)
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        _depth.reset(token)


def compose_wrappers(
    entries: tuple[Registration, ...], hook: str, terminal: Callable
) -> Callable:
    handler = terminal
    for entry in reversed(entries):
        binding = entry.hook(hook)
        if binding is None:
            continue

        def layer(binding, next_handler):
            async def invoke(request):
                count = 0

                async def bounded(req):
                    nonlocal count
                    count += 1
                    if count > 3:
                        raise PluginError("每个 wrapper 最多调用 handler 三次")
                    return await next_handler(req)

                if binding.async_:
                    return await binding.async_(request, bounded)
                return await call_sync_wrapper(binding.sync, request, bounded)

            return invoke

        handler = layer(binding, handler)
    return handler
