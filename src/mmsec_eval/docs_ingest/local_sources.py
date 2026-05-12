# 文件说明：该文件属于资料摄取层，集中实现 local sources 相关逻辑。
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.io.yaml_io import read_yaml


# 中文注释：定义 LocalSource 的结构化职责，作为资料摄取层中状态、配置或行为的边界。
@dataclass
class LocalSource:
    requested_path: str
    resolved_path: str
    exists: bool
    size: int
    sha256: str
    file_type: str
    parser: str
    error: str = ""


# 中文注释：封装 _sha256 的内部步骤，让资料摄取层主流程保持清晰并隔离边界细节。
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# 中文注释：封装 _select_parser 的内部步骤，让资料摄取层主流程保持清晰并隔离边界细节。
def _select_parser(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf", "parse_pdf"
    if ext in {".doc", ".docx"}:
        return "doc", "parse_doc"
    return "text", "parse_text"


# 中文注释：封装 _resolve_one 的内部步骤，让资料摄取层主流程保持清晰并隔离边界细节。
def _resolve_one(raw_path: str, mapping: dict[str, str]) -> Path:
    requested = Path(raw_path)
    if requested.exists():
        return requested

    key = requested.name
    if key in mapping:
        mapped = Path(mapping[key])
        if mapped.exists():
            return mapped

    # Secondary lookup by stem to avoid filename suffix mismatches.
    stem = requested.stem
    for mk, mv in mapping.items():
        if Path(mk).stem == stem:
            mapped = Path(mv)
            if mapped.exists():
                return mapped

    default_path = Path("assets/papers") / key
    return default_path


# 中文注释：实现 resolve_local_sources 的核心流程，支撑资料摄取层中的业务语义和异常边界。
def resolve_local_sources(cfg: AppConfig) -> list[LocalSource]:
    map_path = Path(cfg.docs.local_paths_file)
    mapping_data = read_yaml(str(map_path)) if map_path.exists() else {}
    mapping = mapping_data.get("sources", {}) if isinstance(mapping_data, dict) else {}

    out: list[LocalSource] = []
    for raw in cfg.docs.paths:
        resolved = _resolve_one(raw, mapping)
        file_type, parser = _select_parser(resolved)
        exists = resolved.exists()
        size = resolved.stat().st_size if exists else 0
        sha = _sha256(resolved) if exists else ""
        out.append(
            LocalSource(
                requested_path=raw,
                resolved_path=str(resolved),
                exists=exists,
                size=size,
                sha256=sha,
                file_type=file_type,
                parser=parser,
                error="" if exists else "file_not_found",
            )
        )
    return out


# 中文注释：实现 local_source_to_dict 的核心流程，支撑资料摄取层中的业务语义和异常边界。
def local_source_to_dict(src: LocalSource) -> dict[str, Any]:
    return asdict(src)

