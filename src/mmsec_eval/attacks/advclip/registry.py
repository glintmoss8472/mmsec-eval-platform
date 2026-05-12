from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def registry_path(artifacts_dir: str) -> Path:
    return Path(artifacts_dir) / "advclip_patch_registry.json"


def make_key(*, clip_model_name: str, mode: str, patch_size: int) -> str:
    return f"{clip_model_name}:{str(mode).upper()}:{int(patch_size)}"


def _empty_registry() -> dict[str, Any]:
    return {"version": 1, "entries": {}}


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


def write_registry_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _relativize_under_artifacts(artifacts_dir: str, patch_path: str) -> str:
    art = Path(artifacts_dir).resolve()
    p = Path(patch_path)
    try:
        rel = p.resolve().relative_to(art)
        return rel.as_posix()
    except (OSError, ValueError):
        # Keep original path (may be absolute or already relative).
        return str(patch_path)


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


@dataclass(frozen=True)
class RegistryEntry:
    key: str
    patch_path: str
    run_id: str
    trained: bool
    use_gan: bool
    updated_at: str


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
