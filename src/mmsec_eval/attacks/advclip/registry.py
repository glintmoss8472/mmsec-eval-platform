# 文件说明：该文件属于AdvCLIP 攻击模块，集中实现 registry 相关逻辑。
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 执行 `utc now iso` 辅助逻辑，保持AdvCLIP 攻击模块中的输入处理和结果输出一致。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# 定位 `registry 路径`，把配置值或请求上下文转换成实际文件系统路径。
def registry_path(artifacts_dir: str) -> Path:
    return Path(artifacts_dir) / "advclip_patch_registry.json"


# 构建 `key` 数据，集中整理AdvCLIP 攻击模块需要的输出结构。
def make_key(*, clip_model_name: str, mode: str, patch_size: int) -> str:
    return f"{clip_model_name}:{str(mode).upper()}:{int(patch_size)}"


# 执行 `empty registry` 辅助逻辑，保持AdvCLIP 攻击模块中的输入处理和结果输出一致。
def _empty_registry() -> dict[str, Any]:
    return {"version": 1, "entries": {}}


# 读取 `registry`，并对缺失或异常输入做边界处理。
def read_registry(artifacts_dir: str) -> dict[str, Any]:
    p = registry_path(artifacts_dir)
    if not p.exists():
        return _empty_registry()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty_registry()
        if "entries" not in data or not isinstance(data.get("entries"), dict):
            data["entries"] = {}
        if "version" not in data:
            data["version"] = 1
        return data
    except (json.JSONDecodeError, OSError):
        return _empty_registry()


# 写出 `registry atomic`，保证后续报告、页面或复现实验能读取。
def write_registry_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# 执行 `relativize under 产物` 辅助逻辑，保持AdvCLIP 攻击模块中的输入处理和结果输出一致。
def _relativize_under_artifacts(artifacts_dir: str, patch_path: str) -> str:
    art = Path(artifacts_dir).resolve()
    p = Path(patch_path)
    try:
        rel = p.resolve().relative_to(art)
        return rel.as_posix()
    except (OSError, ValueError):
        # Keep original path (may be absolute or already relative).
        return str(patch_path)


# 更新 `entry`，把最新状态同步到存储、页面或运行上下文。
def update_entry(
    *,
    artifacts_dir: str,
    key: str,
    patch_path: str,
    run_id: str,
    trained: bool,
    use_gan: bool,
) -> None:
    data = read_registry(artifacts_dir)
    entries = data.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        data["entries"] = entries

    entries[str(key)] = {
        "patch_path": _relativize_under_artifacts(artifacts_dir, patch_path),
        "run_id": str(run_id),
        "trained": bool(trained),
        "use_gan": bool(use_gan),
        "updated_at": _utc_now_iso(),
    }

    write_registry_atomic(registry_path(artifacts_dir), data)


# 解析 `补丁` 的真实位置或配置值，减少调用方重复分支。
def resolve_patch(artifacts_dir: str, key: str) -> str:
    data = read_registry(artifacts_dir)
    entry = (data.get("entries") or {}).get(str(key), {})
    if not isinstance(entry, dict):
        return ""
    patch_path = str(entry.get("patch_path") or "")
    if not patch_path:
        return ""
    p = Path(patch_path)
    if not p.is_absolute():
        p = Path(artifacts_dir) / p
    return str(p) if p.exists() else ""


# 定义 `RegistryEntry` 的状态和行为边界，供AdvCLIP 攻击模块在固定职责内复用。
@dataclass(frozen=True)
class RegistryEntry:
    key: str
    patch_path: str
    run_id: str
    trained: bool
    use_gan: bool
    updated_at: str


# 获取 `entry`，封装存储查询或状态读取细节。
def get_entry(artifacts_dir: str, key: str) -> RegistryEntry | None:
    data = read_registry(artifacts_dir)
    entry = (data.get("entries") or {}).get(str(key), {})
    if not isinstance(entry, dict):
        return None
    patch_path = str(entry.get("patch_path") or "")
    run_id = str(entry.get("run_id") or "")
    updated_at = str(entry.get("updated_at") or "")
    trained = bool(entry.get("trained", False))
    use_gan = bool(entry.get("use_gan", False))
    if not patch_path:
        return None
    return RegistryEntry(
        key=str(key),
        patch_path=patch_path,
        run_id=run_id,
        trained=trained,
        use_gan=use_gan,
        updated_at=updated_at,
    )
