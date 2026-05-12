from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict

from mmsec_eval.exceptions import PluginNotFoundError

Registry = Dict[str, Dict[str, Callable[[], Any]]]
_REGISTRY: Registry = defaultdict(dict)


def register(kind: str, name: str, factory: Callable[[], Any]) -> None:
    _REGISTRY[kind][name] = factory


def create(kind: str, name: str) -> Any:
    if kind not in _REGISTRY or name not in _REGISTRY[kind]:
        raise PluginNotFoundError(f"plugin not found: {kind}.{name}")
    return _REGISTRY[kind][name]()


def list_plugins(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}).keys())
