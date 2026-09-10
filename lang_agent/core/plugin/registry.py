"""按 agent 隔离的事务注册表及稳定依赖排序。"""

import copy
import hashlib
import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any

from .base import PluginBase
from .types import HOOKS, PluginError


@dataclass(frozen=True, slots=True)
class HookBinding:
    name: str
    sync: Callable[..., Any] | None
    async_: Callable[..., Any] | None


@dataclass(frozen=True, slots=True)
class Registration:
    name: str
    parents: tuple[str, ...]
    version: str
    fingerprint: str
    bindings: tuple[HookBinding, ...]
    buffered: bool
    config_json: str
    validate_answer: Callable[[str, Any, Any], None]

    def hook(self, name: str) -> HookBinding | None:
        return next((b for b in self.bindings if b.name == name), None)


@dataclass(frozen=True, slots=True)
class PluginSpecSnapshot:
    agent_id: str
    registrations: tuple[Registration, ...]

    @property
    def revision(self) -> str:
        return hashlib.sha256(
            json.dumps([(r.name, r.fingerprint) for r in self.registrations]).encode()
        ).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """导出可持久化定义；插件对象与服务句柄不序列化。"""
        return {
            "agent_id": self.agent_id,
            "revision": self.revision,
            "plugins": [
                {
                    "name": r.name,
                    "version": r.version,
                    "config": json.loads(r.config_json),
                    "hooks": [b.name for b in r.bindings],
                    "fingerprint": r.fingerprint,
                }
                for r in self.registrations
            ],
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _registration(plugin: PluginBase, hooks: Sequence[str]) -> Registration:
    if not plugin.name or not isinstance(plugin.name, str):
        raise PluginError("插件 name 不能为空")
    # 保留外部服务/日志句柄，隔离声明配置，调用方后续编辑不影响旧版本。
    plugin = copy.copy(plugin)
    plugin.config = copy.deepcopy(dict(plugin.config))
    plugin.parent_plugins = tuple(plugin.parent_plugins)
    if len(hooks) != len(set(hooks)):
        raise PluginError("重复 hook 名称")
    names = []
    for raw in hooks:
        name = raw[1:] if raw.startswith("a") and raw[1:] in HOOKS else raw
        if name not in HOOKS:
            raise PluginError(f"未知 hook: {raw}")
        if name not in names:
            names.append(name)
    bindings = []
    sources = []
    try:
        sources.append(inspect.getsource(plugin.validate_answer))
    except (OSError, TypeError):
        sources.append(plugin.validate_answer.__qualname__)
    for name in names:
        implementations: list[Callable[..., Any] | None] = []
        for method_name in (name, "a" + name):
            method = getattr(plugin, method_name)
            implemented = getattr(type(plugin), method_name) is not getattr(
                PluginBase, method_name
            )
            if not implemented:
                if (
                    method_name in hooks
                    and method_name.startswith("a")
                    and method_name[1:] in HOOKS
                ):
                    raise PluginError(f"{plugin.name}.{method_name} 未实现")
                implementations.append(None)
                continue
            try:
                inspect.signature(method).bind(None, None)
            except TypeError as exc:
                raise PluginError(f"{plugin.name}.{method_name} 签名错误") from exc
            if inspect.iscoroutinefunction(method) != (method_name == "a" + name):
                raise PluginError(f"{plugin.name}.{method_name} 同步/异步定义错误")
            implementations.append(method)
            try:
                sources.append(inspect.getsource(method))
            except (OSError, TypeError):
                sources.append(f"{method.__module__}.{method.__qualname__}")
        if not any(implementations):
            raise PluginError(f"{plugin.name}.{name} 未实现")
        bindings.append(HookBinding(name, *implementations))
    manifest = {
        "name": plugin.name,
        "version": plugin.version,
        "parents": list(plugin.parent_plugins),
        "config": plugin.config,
        "hooks": names,
        "source": sources,
        "buffered": plugin.requires_buffered_output,
    }
    try:
        fingerprint = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise PluginError("插件 config 必须可 JSON 序列化") from exc
    config_json = json.dumps(plugin.config, sort_keys=True, allow_nan=False)
    plugin.config = _freeze(plugin.config)
    return Registration(
        plugin.name,
        tuple(plugin.parent_plugins),
        plugin.version,
        fingerprint,
        tuple(bindings),
        plugin.requires_buffered_output,
        config_json,
        plugin.validate_answer,
    )


def _ordered(entries: dict[str, Registration]) -> tuple[Registration, ...]:
    for entry in entries.values():
        missing = set(entry.parents) - entries.keys()
        if missing:
            raise PluginError(f"{entry.name} 缺少依赖: {sorted(missing)}")
    result: list[Registration] = []
    pending = dict(entries)
    while pending:
        ready = next(
            (
                e
                for e in pending.values()
                if all(p in {r.name for r in result} for p in e.parents)
            ),
            None,
        )
        if ready is None:
            raise PluginError(f"插件依赖成环: {list(pending)}")
        result.append(ready)
        del pending[ready.name]
    return tuple(result)


class PluginRegistry:
    """编辑注册表不发布运行配置；调用 loop.update_plugin_hooks 显式发布。"""

    def __init__(self) -> None:
        self._agents: dict[str, dict[str, Registration]] = {}
        self._lock = RLock()

    def register(
        self, plugin: PluginBase, *, agent_id: str, hooks: Sequence[str]
    ) -> None:
        self._change(plugin, agent_id, hooks, replace=False)

    def replace(
        self, name: str, plugin: PluginBase, *, agent_id: str, hooks: Sequence[str]
    ) -> None:
        if name != plugin.name:
            raise PluginError("replace 不能改变插件名称")
        self._change(plugin, agent_id, hooks, replace=True)

    def _change(
        self, plugin: PluginBase, agent_id: str, hooks: Sequence[str], *, replace: bool
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id:
            raise PluginError("agent_id 不能为空")
        entry = _registration(plugin, hooks)
        with self._lock:
            entries = dict(self._agents.get(agent_id, {}))
            if (entry.name in entries) != replace:
                raise PluginError(f"插件重复注册或替换目标不存在: {entry.name}")
            entries[entry.name] = entry
            _ordered(entries)
            self._agents[agent_id] = entries

    def unregister(self, name: str, *, agent_id: str) -> None:
        with self._lock:
            entries = dict(self._agents.get(agent_id, {}))
            if name not in entries:
                raise PluginError(f"插件不存在: {name}")
            del entries[name]
            _ordered(entries)
            self._agents[agent_id] = entries

    def snapshot(self, *, agent_id: str) -> PluginSpecSnapshot:
        if not agent_id:
            raise PluginError("agent_id 不能为空")
        with self._lock:
            return PluginSpecSnapshot(
                agent_id, _ordered(self._agents.get(agent_id, {}))
            )
