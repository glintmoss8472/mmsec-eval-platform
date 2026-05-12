# 文件说明：该文件属于运维与实验脚本，集中实现 build artifacts manifest 相关逻辑。
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_FILES = (
    "summary.json",
    "report_data.json",
    "report.html",
    "results.jsonl",
    "case_bundle.json",
    "cases_index.jsonl",
    "config_snapshot.json",
    "env_snapshot.json",
)


# 执行 `rel` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


# 执行 `sha256` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# 读取 `JSON`，并对缺失或异常输入做边界处理。
def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


# 执行 `file entry` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _rel(path, root),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


# 执行 `scan 运行记录` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def scan_run(run_dir: Path, root: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    files = {name: _file_entry(run_dir / name, root) for name in KEY_FILES if (run_dir / name).exists()}
    debug_dir = run_dir / "attack_debug"
    preview_dir = run_dir / "samples_preview"
    return {
        "run_id": run_dir.name,
        "run_dir": _rel(run_dir, root),
        "dataset": summary.get("dataset") or summary.get("dataset_kind") or summary.get("dataset_name", ""),
        "attack": summary.get("attack") or summary.get("attack_name", ""),
        "model": summary.get("model_adapter") or summary.get("model", ""),
        "task_kind": summary.get("task_kind", ""),
        "risk_score": summary.get("risk_score", 0.0),
        "files": files,
        "has_attack_debug": debug_dir.exists(),
        "has_sample_preview": preview_dir.exists(),
    }


# 整理 `scan paper rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def scan_paper_rows(paper_root: Path, root: Path) -> list[dict[str, Any]]:
    rows_dir = paper_root / "rows"
    if not rows_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row_dir in sorted(item for item in rows_dir.iterdir() if item.is_dir()):
        row_files = {
            name: _file_entry(row_dir / name, root)
            for name in ("portable_summary.json", "portable_report_data.json", "portable_report.html", "row_evidence.json")
            if (row_dir / name).exists()
        }
        rows.append({"row_id": row_dir.name, "row_dir": _rel(row_dir, root), "files": row_files})
    return rows


# 执行 `scan screenshots` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def scan_screenshots(root: Path) -> list[dict[str, Any]]:
    screenshot_dir = root / "docs" / "assets" / "server_ui_content_audit_20260419"
    if not screenshot_dir.exists():
        return []
    return [_file_entry(path, root) for path in sorted(screenshot_dir.glob("*.png"))]


# 执行 `scan auxiliary 产物` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def scan_auxiliary_artifacts(artifacts_dir: Path, root: Path) -> list[dict[str, Any]]:
    patterns = {
        "embodied_decision": ("embodied_decision*", ("decision_summary.json", "decision_results.jsonl")),
        "vqa_dialogue": ("vqa_dialogue*", ("interaction_summary.json", "interaction_results.jsonl")),
        "asset_check": ("*asset_check*", ("*.json",)),
        "strict_protocol": ("strict_paper_protocol*", ("strict_paper_reproduction_audit.md", "audit.json")),
    }
    groups: list[dict[str, Any]] = []
    for group_name, (dir_pattern, file_patterns) in patterns.items():
        for directory in sorted(path for path in artifacts_dir.glob(dir_pattern) if path.is_dir()):
            files: list[dict[str, Any]] = []
            for file_pattern in file_patterns:
                files.extend(_file_entry(path, root) for path in sorted(directory.glob(file_pattern)) if path.is_file())
            groups.append({"group": group_name, "dir": _rel(directory, root), "files": files})
    return groups


# 构建 `manifest` 数据，集中整理运维与实验脚本需要的输出结构。
def build_manifest(project_root: Path, artifacts_dir: Path) -> dict[str, Any]:
    runs_dir = artifacts_dir / "runs"
    paper_root = artifacts_dir / "paper_suite_20260418_final"
    runs = [scan_run(path, project_root) for path in sorted(runs_dir.iterdir()) if path.is_dir()] if runs_dir.exists() else []
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root.resolve()),
        "artifacts_dir": _rel(artifacts_dir, project_root),
        "run_count": len(runs),
        "runs": runs,
        "paper_suite": {
            "root": _rel(paper_root, project_root),
            "exists": paper_root.exists(),
            "rows": scan_paper_rows(paper_root, project_root),
        },
        "screenshots": scan_screenshots(project_root),
        "auxiliary_artifacts": scan_auxiliary_artifacts(artifacts_dir, project_root),
    }


# 解析 `args`，把文本或载荷转换成可校验的字段。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducibility manifest for thesis and experiment artifacts.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--out", default="artifacts/manifest.json")
    return parser.parse_args()


# 作为 `build_artifacts_manifest.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    artifacts_dir = (root / args.artifacts_dir).resolve()
    payload = build_manifest(root, artifacts_dir)
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "out": _rel(out_path, root), "run_count": payload["run_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
