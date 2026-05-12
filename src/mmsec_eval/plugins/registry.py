# 文件说明：该文件属于项目工程，集中实现 registry 相关逻辑。
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict

from mmsec_eval.exceptions import PluginNotFoundError

Registry = Dict[str, Dict[str, Callable[[], Any]]]
_REGISTRY: Registry = defaultdict(dict)


# 中文注释：实现 register 的核心流程，支撑项目工程中的业务语义和异常边界。
def register(kind: str, name: str, factory: Callable[[], Any]) -> None:
    _REGISTRY[kind][name] = factory


# 中文注释：实现 create 的核心流程，支撑项目工程中的业务语义和异常边界。
def create(kind: str, name: str) -> Any:
    if kind not in _REGISTRY or name not in _REGISTRY[kind]:
        raise PluginNotFoundError(f"plugin not found: {kind}.{name}")
    return _REGISTRY[kind][name]()


# 中文注释：实现 list_plugins 的核心流程，支撑项目工程中的业务语义和异常边界。
def list_plugins(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}).keys())
