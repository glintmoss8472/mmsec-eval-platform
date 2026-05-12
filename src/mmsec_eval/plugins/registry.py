# 文件说明：该文件属于项目工程，集中实现 registry 相关逻辑。
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict

from mmsec_eval.exceptions import PluginNotFoundError

Registry = Dict[str, Dict[str, Callable[[], Any]]]
_REGISTRY: Registry = defaultdict(dict)


# 执行 `register` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def register(kind: str, name: str, factory: Callable[[], Any]) -> None:
    _REGISTRY[kind][name] = factory


# 执行 `create` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def create(kind: str, name: str) -> Any:
    if kind not in _REGISTRY or name not in _REGISTRY[kind]:
        raise PluginNotFoundError(f"plugin not found: {kind}.{name}")
    return _REGISTRY[kind][name]()


# 列出 `插件`，按调用方需要组织分页或过滤结果。
def list_plugins(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}).keys())
