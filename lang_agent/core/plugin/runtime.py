"""不可变插件版本、按槽位复用与原子发布。"""

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any

from langgraph.types import interrupt

from .graph import HookInvocation, compile_hook, state_delta
from .registry import PluginSpecSnapshot
from .types import (
    HOOKS,
    NODE_HOOKS,
    PluginAgentMismatch,
    PluginError,
    PluginRevisionUnavailable,
)


@dataclass(frozen=True, slots=True)
class CompiledPluginBundle:
    snapshot: PluginSpecSnapshot
    graphs: Mapping[str, Any]

    @property
    def revision(self) -> str:
        return self.snapshot.revision

    @property
    def agent_id(self) -> str:
        return self.snapshot.agent_id

    @property
    def buffered(self) -> bool:
        return any(r.buffered for r in self.snapshot.registrations)


@dataclass(frozen=True, slots=True)
class PluginUpdateResult:
    agent_id: str
    previous_revision: str
    revision: str
    changed_hooks: tuple[str, ...]


def _fingerprint(snapshot: PluginSpecSnapshot, hook: str):
    return tuple(
        (r.name, r.fingerprint) for r in snapshot.registrations if r.hook(hook)
    )


class PluginRuntime:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        empty = CompiledPluginBundle(
            PluginSpecSnapshot(agent_id, ()),
            MappingProxyType(dict.fromkeys(NODE_HOOKS)),
        )
        self._current = empty
        self._versions = {empty.revision: empty}
        self._lock = RLock()

    def get(self, revision: str | None = None) -> CompiledPluginBundle:
        with self._lock:
            if revision is None:
                return self._current
            if revision not in self._versions:
                raise PluginRevisionUnavailable(
                    f"agent {self.agent_id} 缺少插件版本 {revision}"
                )
            return self._versions[revision]

    def update(
        self, snapshot: PluginSpecSnapshot, expected_revision: str | None = None
    ) -> PluginUpdateResult:
        if snapshot.agent_id != self.agent_id:
            raise PluginAgentMismatch("snapshot.agent_id 与 loop.agent_id 不一致")
        old = self.get()
        changed = tuple(
            h
            for h in HOOKS
            if _fingerprint(old.snapshot, h) != _fingerprint(snapshot, h)
        )
        graphs = dict(old.graphs)
        for hook in changed:
            if hook in NODE_HOOKS:
                graphs[hook] = compile_hook(snapshot.registrations, hook)
        bundle = CompiledPluginBundle(snapshot, MappingProxyType(graphs))
        with self._lock:
            expected = (
                expected_revision if expected_revision is not None else old.revision
            )
            if self._current.revision != expected:
                raise PluginError("插件发布版本冲突，请重新获取 snapshot")
            self._versions[bundle.revision] = bundle
            self._current = bundle
        return PluginUpdateResult(self.agent_id, old.revision, bundle.revision, changed)

    def validate(self, state: Any, context: Any) -> CompiledPluginBundle:
        bundle = getattr(context, "plugin_bundle", None)
        if bundle is None:
            if self.get().snapshot.registrations:
                raise PluginError("有插件时必须 bind_plugin_context 并提供版本输入")
            return self.get()
        if bundle.agent_id != self.agent_id or context.agent_id != self.agent_id:
            raise PluginAgentMismatch("context 属于另一个 agent")
        if (
            state.get("agent_id") != self.agent_id
            or state.get("plugin_revision") != bundle.revision
        ):
            raise PluginAgentMismatch(
                "state 的 agent_id/plugin_revision 与 context 不一致"
            )
        return bundle

    def validate_answers(
        self, revision: str, interruptions: list[Any], answers: dict[str, Any]
    ) -> None:
        bundle = self.get(revision)
        entries = {r.name: r for r in bundle.snapshot.registrations}
        for item in interruptions:
            payload = item.value
            if (
                payload.get("agent_id") != self.agent_id
                or payload.get("revision") != revision
            ):
                raise PluginAgentMismatch("审批身份或版本不匹配")
            entry = entries.get(payload.get("plugin"))
            if entry is None:
                raise PluginError("审批插件不存在")
            entry.validate_answer(
                payload["key"],
                copy.deepcopy(payload["payload"]),
                copy.deepcopy(answers[item.id]),
            )

    async def dispatch(
        self, hook: str, state: Any, config: Any, context: Any
    ) -> dict[str, Any]:
        bundle = self.validate(state, context)
        graph = bundle.graphs[hook]
        delta: dict[str, Any] = (
            {"model_route_override": None} if hook == "before_model" else {}
        )
        if graph is None:
            return delta
        answers: dict[str, dict[str, Any]] = {}
        for _ in range(16):
            invocation = HookInvocation(
                context,
                config,
                hook,
                self.agent_id,
                bundle.revision,
                state.get("run_id", ""),
                answers,
            )
            internal_config = {
                **config,
                "recursion_limit": len(bundle.snapshot.registrations) + 4,
            }
            result = await graph.ainvoke(
                {"working": copy.deepcopy(dict(state)), "pause": None, "route": None},
                internal_config,
                context=invocation,
            )
            pause = result.get("pause")
            if not pause:
                delta.update(state_delta(dict(state), result["working"]))
                if result.get("route"):
                    delta["model_route_override"] = result["route"]
                return delta
            payload = {
                **pause,
                "agent_id": self.agent_id,
                "revision": bundle.revision,
                "run_id": state.get("run_id"),
                "hook": hook,
            }
            json.dumps(payload, allow_nan=False)
            if pause["key"] in answers.get(pause["plugin"], {}):
                raise PluginError("hook 重复请求已回答的审批 key")
            answer = interrupt(payload)
            json.dumps(answer, allow_nan=False)
            answers.setdefault(pause["plugin"], {})[pause["key"]] = answer
        raise PluginError("单个 hook 审批次数超过 16 次")
