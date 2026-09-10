"""双函数节点、无持久 checkpoint 的内部图及消息增量合并。"""

import asyncio
import copy
import json
from dataclasses import dataclass
from typing import Any, TypedDict
from uuid import NAMESPACE_URL, uuid5

from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph._internal._runnable import RunnableCallable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from .registry import Registration
from .types import HookResult, HookRuntime, PluginError


@dataclass(frozen=True, slots=True)
class HookInvocation:
    context: Any
    config: RunnableConfig
    hook: str
    agent_id: str
    revision: str
    run_id: str
    answers: dict[str, dict[str, Any]]


class HookGraphState(TypedDict):
    working: dict[str, Any]
    pause: Any
    route: Any


def merge_namespaces(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


def merge_update(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(state)
    for key, value in update.items():
        if key == "messages":
            result[key] = add_messages(state.get(key, []), value)
        elif key == "plugin_state":
            result[key] = merge_namespaces(state.get(key), value)
        else:
            result[key] = value
    return result


def state_delta(original: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """只提交变化；重排时用显式删除重建，以匹配 add_messages。"""
    delta = {k: v for k, v in final.items() if k != "messages" and original.get(k) != v}
    old, new = original.get("messages", []), final.get("messages", [])
    if old != new:
        old_by_id = {m.id: m for m in old}
        new_ids = {m.id for m in new}
        patch: list[Any] = [RemoveMessage(id=m.id) for m in old if m.id not in new_ids]
        patch += [m for m in new if old_by_id.get(m.id) != m]
        if add_messages(copy.deepcopy(old), copy.deepcopy(patch)) != new:
            patch = [RemoveMessage(id=m.id) for m in old] + new
        delta["messages"] = patch
    return delta


def _prepare(
    state: HookGraphState, entry: Registration, runtime: Runtime[HookInvocation]
):
    invocation = runtime.context
    if invocation is None:
        raise PluginError("hook context 缺失")
    info = HookRuntime(
        invocation.agent_id,
        invocation.context,
        invocation.config,
        invocation.hook,
        entry.name,
        invocation.revision,
        invocation.run_id,
        copy.deepcopy(invocation.answers.get(entry.name, {})),
    )
    return copy.deepcopy(state["working"]), info


def _commit(
    state: HookGraphState, info: HookRuntime, result: HookResult | None
) -> dict[str, Any]:
    if result is None:
        return {}
    if not isinstance(result, HookResult):
        raise PluginError(
            f"{info.plugin_name}.{info.hook_name} 必须返回 HookResult 或 None"
        )
    if result.pause:
        if (
            result.update
            or result.route
            or not isinstance(result.pause.key, str)
            or not result.pause.key
        ):
            raise PluginError("pause 不可同时提交 update/route，key 必须非空")
        json.dumps(result.pause.payload, allow_nan=False)
        return {
            "pause": {
                "plugin": info.plugin_name,
                "key": result.pause.key,
                "payload": result.pause.payload,
            }
        }
    if result.route and (
        info.hook_name != "after_model" or result.route not in ("retry", "end")
    ):
        raise PluginError("route 仅允许 after_model 返回 retry/end")
    update = copy.deepcopy(result.update)
    if update.keys() - {"messages", "plugin_state"}:
        raise PluginError("hook update 包含受保护字段")
    namespaces = update.get("plugin_state", {})
    if not isinstance(namespaces, dict) or namespaces.keys() - {info.plugin_name}:
        raise PluginError("plugin_state 只能修改自己的 namespace")
    json.dumps(namespaces, allow_nan=False)
    for index, message in enumerate(update.get("messages", [])):
        if isinstance(message, AIMessageChunk):
            raise PluginError("hook 不允许向 checkpoint 写入 AIMessageChunk")
        if not hasattr(message, "id"):
            raise PluginError("messages 必须使用消息对象")
        if message.id is None:
            seed = f"{info.agent_id}:{info.run_id}:{info.hook_name}:{info.plugin_name}:{len(state['working'].get('messages', []))}:{index}"
            message.id = uuid5(NAMESPACE_URL, seed).hex
    working = merge_update(state["working"], update)
    if result.route:
        messages = working.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if last_ai and (last_ai.tool_calls or last_ai.invalid_tool_calls):
            raise PluginError("强制路由前需修正当前 AIMessage 的工具调用")
    return {"working": working, "route": result.route}


def build_hook_node(entry: Registration, hook: str) -> RunnableCallable:
    binding = entry.hook(hook)
    assert binding is not None

    def sync_node(
        state: HookGraphState, config: RunnableConfig, runtime: Runtime[HookInvocation]
    ):
        working, info = _prepare(state, entry, runtime)
        assert binding.sync is not None
        return _commit(state, info, binding.sync(working, info))

    async def async_node(
        state: HookGraphState, config: RunnableConfig, runtime: Runtime[HookInvocation]
    ):
        if binding.async_ is None:
            return await asyncio.to_thread(sync_node, state, config, runtime)
        working, info = _prepare(state, entry, runtime)
        return _commit(state, info, await binding.async_(working, info))

    return RunnableCallable(
        sync_node if binding.sync else None, async_node, trace=False
    )


def compile_hook(entries: tuple[Registration, ...], hook: str) -> Any:
    selected = [e for e in entries if e.hook(hook)]
    if not selected:
        return None
    builder = StateGraph(HookGraphState, context_schema=HookInvocation)
    for index, entry in enumerate(selected):
        name = f"{entry.name}.{hook}"
        builder.add_node(name, build_hook_node(entry, hook))
        next_name = (
            f"{selected[index + 1].name}.{hook}" if index + 1 < len(selected) else END
        )

        def route(state, destination=next_name):
            return END if state.get("pause") or state.get("route") else destination

        builder.add_conditional_edges(name, route, list({next_name, END}))
    builder.add_edge(START, f"{selected[0].name}.{hook}")
    return builder.compile(checkpointer=False)
