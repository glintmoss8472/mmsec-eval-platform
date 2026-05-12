from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.io.yaml_io import read_yaml


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_parser(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf", "parse_pdf"
    if ext in {".doc", ".docx"}:
        return "doc", "parse_doc"
    return "text", "parse_text"


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


def local_source_to_dict(src: LocalSource) -> dict[str, Any]:
    return asdict(src)

