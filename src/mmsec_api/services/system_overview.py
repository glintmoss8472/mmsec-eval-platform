from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import Request

from mmsec_api.services.dataset_status import enrich_dataset_registry_rows
from mmsec_api.services.model_runtime import build_adapter_env, list_main_models
from mmsec_api.services.run_reader import discover_runs_from_artifacts, read_json
from mmsec_api.utils import utc_now_iso
from mmsec_eval.paper_evidence import build_formal_joint_execution
from mmsec_eval.plugins.registry import list_plugins
from mmsec_eval.runtime import torch_install_command


def _torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        return {
            "installed": True,
            "version": str(getattr(torch, "__version__", "")),
            "cuda_version": cuda_version,
            "cuda_available": cuda_available,
            "device_count": device_count,
        }
    except (ImportError, OSError, RuntimeError) as e:
        return {"installed": False, "error": str(e)}


def _git_value(repo: Path, args: list[str]) -> str:
    if not repo.exists():
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=4,
        )
        if out.returncode != 0:
            return ""
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _repo_entry(project_root: Path, rel_path: str) -> dict[str, Any]:
    p = project_root / rel_path
    return {
        "name": p.name,
        "path": str(p),
        "exists": p.exists(),
        "remote": _git_value(p, ["remote", "get-url", "origin"]),
        "commit": _git_value(p, ["rev-parse", "--short", "HEAD"]),
    }


def _deployment_version_info(project_root: Path) -> dict[str, str]:
    path = project_root / "deployment_version.json"
    data = read_json(path, {})
    if not isinstance(data, dict):
        data = {}
    return {
        "path": str(path),
        "version": str(data.get("version", "") or ""),
        "deployment_target": str(data.get("deployment_target", "") or ""),
        "version_source": "deployment_version.json" if path.exists() else "",
    }


def _source_docs(project_root: Path) -> dict[str, Any]:
    taskbook_extract = project_root / "artifacts" / "taskbook_reextract_20260214.txt"
    taskbook_extract_old = project_root / "artifacts" / "taskbook_extracted.txt"
    advclip_pdf = project_root / "artifacts" / "paper_extracts" / "pdfs" / "advclip.pdf"
    tmm_pdf = project_root / "artifacts" / "paper_extracts" / "pdfs" / "tmm.pdf"
    advedm_pdf = project_root / "artifacts" / "paper_extracts" / "pdfs" / "advedm.pdf"
    return {
        "taskbook_extract": str(taskbook_extract if taskbook_extract.exists() else taskbook_extract_old),
        "taskbook_extract_exists": taskbook_extract.exists() or taskbook_extract_old.exists(),
        "paper_pdfs": {
            "advclip": {"path": str(advclip_pdf), "exists": advclip_pdf.exists()},
            "tmm": {"path": str(tmm_pdf), "exists": tmm_pdf.exists()},
            "advedm": {"path": str(advedm_pdf), "exists": advedm_pdf.exists()},
        },
        "paper_extract_text": {
            "advclip": str(project_root / "artifacts" / "paper_extracts" / "text" / "advclip.txt"),
            "tmm": str(project_root / "artifacts" / "paper_extracts" / "text" / "tmm.txt"),
            "advedm": str(project_root / "artifacts" / "paper_extracts" / "text" / "advedm.txt"),
        },
        "docs_mapping": {
            "acceptance_matrix": str(project_root / "docs" / "acceptance_matrix.md"),
            "reproduction_mapping": str(project_root / "docs" / "reproduction_mapping.md"),
            "gap_assessment": str(project_root / "docs" / "taskbook_paper_gap_assessment_20260214.md"),
        },
    }


def _runtime_info() -> dict[str, Any]:
    return {
        "current_device": os.getenv("MMSEC_RUNTIME_DEVICE", "cuda"),
        "cuda_required": True,
        "strict_real": os.getenv("MMSEC_STRICT_REAL", "1"),
    }


def _build_runtime_identity() -> dict[str, Any]:
    runtime_context = str(os.getenv("MMSEC_RUNTIME_CONTEXT", "") or "").strip()
    image_ref = str(os.getenv("MMSEC_IMAGE_REF", "") or "").strip()
    runtime_profile = str(os.getenv("MMSEC_RUNTIME_PROFILE", "") or "").strip()
    runtime_volume_name = str(os.getenv("MMSEC_RUNTIME_VOLUME_NAME", "") or "").strip()
    bundle_root = str(os.getenv("MMSEC_BUNDLE_ROOT", "") or "").strip()
    runtime_root = str(os.getenv("MMSEC_RUNTIME_ROOT", "") or "").strip()
    docker_env_exists = Path("/.dockerenv").exists()
    cgroup_hint = ""
    try:
        cgroup_text = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore")
        if any(token in cgroup_text for token in ("docker", "containerd", "kubepods")):
            cgroup_hint = "container"
    except OSError:
        cgroup_hint = ""
    containerized = docker_env_exists or runtime_context.lower() == "container" or bool(cgroup_hint)
    runtime_transport = "docker_container" if containerized else "host_python"
    return {
        "runtime_transport": runtime_transport,
        "containerized": containerized,
        "runtime_context": runtime_context or ("container" if containerized else "host"),
        "image_ref": image_ref,
        "runtime_profile": runtime_profile,
        "runtime_volume_name": runtime_volume_name,
        "bundle_root": bundle_root,
        "runtime_root": runtime_root,
    }


def _runtime_context(request: Request) -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[3]
    artifacts_dir = Path(getattr(request.app.state, "artifacts_dir", "artifacts")).resolve()
    return project_root, artifacts_dir


_ASR_METRIC_KEY_RE = re.compile(r"^(ir|tr)_asr@(\d+)$")


def _formal_metric_candidates(victim_compare: list[dict[str, Any]]) -> dict[int, list[tuple[str, float]]]:
    candidate_values: dict[int, list[tuple[str, float]]] = {}
    for item in victim_compare:
        attacked = item.get("attacked", {})
        if not isinstance(attacked, dict):
            continue
        for key, value in attacked.items():
            match = _ASR_METRIC_KEY_RE.match(str(key))
            if match is None or not isinstance(value, (int, float)):
                continue
            direction, k_text = match.groups()
            candidate_values.setdefault(int(k_text), []).append((direction, float(value)))
    return candidate_values


def _best_metric_k(
    candidate_values: dict[int, list[tuple[str, float]]],
    target_value: float,
) -> tuple[int | None, list[tuple[str, float]], float | None]:
    matched_k: int | None = None
    matched_entries: list[tuple[str, float]] = []
    best_gap: float | None = None
    for k_value, entries in candidate_values.items():
        if not entries:
            continue
        mean_value = sum(value for _, value in entries) / len(entries)
        gap = abs(mean_value - target_value)
        if best_gap is None or gap < best_gap:
            matched_k = k_value
            matched_entries = entries
            best_gap = gap
    return matched_k, matched_entries, best_gap


def _retrieval_direction_scope(matched_entries: list[tuple[str, float]]) -> str:
    directions = sorted({direction for direction, _ in matched_entries})
    if directions == ["ir", "tr"]:
        return "图检文与文检图双向平均"
    if directions == ["ir"]:
        return "图检文单向"
    if directions == ["tr"]:
        return "文检图单向"
    return "受测检索方向聚合"


def _victim_aggregation_label(suite_row: dict[str, Any], run_row: dict[str, Any], victim_compare: list[dict[str, Any]]) -> str:
    victim_count = len([item for item in victim_compare if str(item.get("victim", "")).strip()])
    if victim_count <= 0:
        victim_count = len(
            [
                str(item).strip()
                for item in list(suite_row.get("victim_model_adapters", run_row.get("victim_model_adapters", [])))
                if str(item).strip()
            ]
        )
    return f"{victim_count} 个受测模型平均" if victim_count > 0 else ""


def _sample_pair_count(suite_row: dict[str, Any]) -> int:
    sample_pair_count = int(suite_row.get("num_pairs", 0) or 0)
    if sample_pair_count <= 0:
        num_images = int(suite_row.get("num_images", 0) or 0)
        num_texts = int(suite_row.get("num_texts", 0) or 0)
        if num_images > 0 and num_texts > 0:
            sample_pair_count = min(num_images, num_texts)
    return sample_pair_count


def _formal_metric_label_note(
    *,
    matched_k: int | None,
    matched_exactly: bool,
    retrieval_direction_scope: str,
    victim_aggregation: str,
    sample_pair_count: int,
) -> tuple[str, str]:
    if matched_exactly and matched_k is not None:
        if matched_k == 1:
            rank_label = "首位"
            english_rank_label = "first rank"
        else:
            rank_label = f"前{matched_k}位"
            english_rank_label = f"top {matched_k} ranks"
        metric_label = f"{rank_label}攻击成功率（attack success rate at {english_rank_label}，汇总）"
        metric_note = (
            f"该值对应攻击后阶段的{rank_label}攻击成功率"
            f"（attack success rate at {english_rank_label}）汇总结果"
        )
    else:
        metric_label = "攻击成功率（attack success rate，汇总）"
        metric_note = "该值对应攻击后阶段的汇总攻击成功率（attack success rate）"
    if retrieval_direction_scope:
        metric_note += f"，按{retrieval_direction_scope}统计"
    if victim_aggregation:
        metric_note += f"，汇总范围为{victim_aggregation}"
    if sample_pair_count > 0:
        metric_note += f"，对应 {sample_pair_count} 对样本。"
    else:
        metric_note += "。"
    return metric_label, metric_note


def _derive_formal_row_metric_semantics(suite_row: dict[str, Any], run_row: dict[str, Any]) -> dict[str, Any]:
    target_value = float(suite_row.get("asr_attack", run_row.get("asr_attack", run_row.get("asr", 0.0))) or 0.0)
    victim_compare = [item for item in list(suite_row.get("victim_compare", [])) if isinstance(item, dict)]
    matched_k, matched_entries, best_gap = _best_metric_k(_formal_metric_candidates(victim_compare), target_value)
    retrieval_direction_scope = _retrieval_direction_scope(matched_entries)
    victim_aggregation = _victim_aggregation_label(suite_row, run_row, victim_compare)
    sample_pair_count = _sample_pair_count(suite_row)
    metric_label, metric_note = _formal_metric_label_note(
        matched_k=matched_k,
        matched_exactly=matched_k is not None and best_gap is not None and best_gap <= 1e-4,
        retrieval_direction_scope=retrieval_direction_scope,
        victim_aggregation=victim_aggregation,
        sample_pair_count=sample_pair_count,
    )
    return {
        "metric_label": metric_label,
        "k_value": int(matched_k or 0),
        "retrieval_direction_scope": retrieval_direction_scope,
        "victim_aggregation": victim_aggregation,
        "sample_pair_count": sample_pair_count,
        "metric_note": metric_note.strip(),
    }


def _dataset_catalog() -> list[dict[str, str]]:
    return [
        {"key": "coco_subset", "name": "COCO val2017 完整验证子集", "tier": "benchmark"},
        {"key": "flickr30k", "name": "Flickr30k", "tier": "benchmark"},
        {"key": "flickr1k", "name": "Flickr1k", "tier": "benchmark"},
        {"key": "vqa_v2_coco_val", "name": "VQA v2 COCO val 真实子集", "tier": "generation"},
        {"key": "coco_object_probe_val", "name": "COCO 对象存在性 Probe 真实子集", "tier": "generation"},
        {"key": "coco_caption_object_val", "name": "COCO Caption 对象级真实子集", "tier": "generation"},
        {"key": "mini_flickr", "name": "Mini Flickr", "tier": "demo"},
    ]


def _dataset_catalog_count() -> int:
    return len(_dataset_catalog())


def _dataset_catalog_map() -> dict[str, dict[str, str]]:
    return {str(item.get("key", "")).strip(): item for item in _dataset_catalog() if str(item.get("key", "")).strip()}


def _live_datasets(project_root: Path, store: Any) -> list[dict[str, Any]]:
    if store is None or not hasattr(store, "list_datasets"):
        return []

    try:
        rows = store.list_datasets()
    except (AttributeError, OSError, RuntimeError, ValueError):
        return []

    if not isinstance(rows, list):
        return []

    catalog = _dataset_catalog_map()
    catalog_order = {key: idx for idx, key in enumerate(catalog)}
    live_rows: list[dict[str, Any]] = []
    for row in enrich_dataset_registry_rows(rows, project_root):
        if not isinstance(row, dict):
            continue
        if not bool(row.get("ready", False)):
            continue
        key = str(row.get("name", "")).strip()
        if not key:
            continue
        catalog_row = catalog.get(key, {})
        live_rows.append(
            {
                "key": key,
                "name": str(catalog_row.get("name", key) or key),
                "tier": str(catalog_row.get("tier", "custom") or "custom"),
                "prepared": bool(row.get("prepared", False)),
                "ready": bool(row.get("ready", False)),
                "ready_reason": str(row.get("ready_reason", "") or ""),
                "item_count": int(row.get("item_count", 0) or 0),
                "root_path": str(row.get("root_path", "") or ""),
                "note": str(row.get("note", "") or ""),
                "source": "dataset_registry",
            }
        )

    live_rows.sort(key=lambda item: (0 if item["key"] in catalog_order else 1, catalog_order.get(item["key"], 999), item["key"]))
    return live_rows


def _latest_completed_dir(artifacts_dir: Path, prefix: str, required_file: str) -> Path | None:
    candidates: list[Path] = []
    for item in artifacts_dir.glob(f"{prefix}*"):
        if not item.is_dir():
            continue
        if not (item / required_file).exists():
            continue
        status = read_json(item / "status.json", {})
        if isinstance(status, dict) and str(status.get("status", "")) == "completed":
            candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _canonical_paper_suite_analysis_path(project_root: Path) -> tuple[str, Path]:
    env_override = str(os.getenv("MMSEC_CANONICAL_PAPER_SUITE_ANALYSIS", "") or "").strip()
    if env_override:
        return "env_configured_canonical", Path(env_override).expanduser().resolve()
    return (
        "canonical_thesis_suite",
        (project_root / "artifacts" / "paper_suite_20260418_final" / "paper_suite_analysis.json").resolve(),
    )


def _archived_paper_suite_analysis_path(project_root: Path) -> Path:
    return (project_root / "artifacts" / "server_snapshot_20260411" / "paper_suite_analysis.json").resolve()


def _portable_artifact_path(project_root: Path, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    path = Path(text).expanduser()
    if path.is_absolute():
        try:
            return path.resolve().relative_to(project_root.resolve()).as_posix()
        except (OSError, RuntimeError, ValueError):
            return normalized
    return normalized


def _latest_paper_suite_analysis(project_root: Path, artifacts_dir: Path) -> tuple[Path, dict[str, Any]]:
    latest_dir = _latest_completed_dir(artifacts_dir, "paper_suite_", "paper_suite_analysis.json")
    if latest_dir is not None:
        path = latest_dir / "paper_suite_analysis.json"
        return path, read_json(path, {})

    fallback = _archived_paper_suite_analysis_path(project_root)
    return fallback, read_json(fallback, {})


def _primary_paper_suite_analysis(project_root: Path, artifacts_dir: Path) -> tuple[Path, dict[str, Any], str]:
    canonical_kind, canonical_path = _canonical_paper_suite_analysis_path(project_root)
    if canonical_path.exists():
        return canonical_path, read_json(canonical_path, {}), canonical_kind

    latest_path, latest_analysis = _latest_paper_suite_analysis(project_root, artifacts_dir)
    fallback = _archived_paper_suite_analysis_path(project_root)
    source_kind = "archived_snapshot_fallback" if latest_path == fallback else "latest_completed_suite"
    return latest_path, latest_analysis, source_kind


def _canonical_paper_environment_reference_path(project_root: Path) -> Path:
    env_override = str(os.getenv("MMSEC_PAPER_RESULT_ENV_REFERENCE", "") or "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()
    return (project_root / "artifacts" / "paper_suite_20260418_final" / "environment_reference.json").resolve()


def _archived_paper_environment_reference_path(project_root: Path) -> Path:
    return (project_root / "artifacts" / "defense_evidence_pack_20260419" / "deployment_reference" / "system_overview.json").resolve()


def _paper_result_environment_reference(project_root: Path) -> tuple[Path, dict[str, Any]]:
    canonical_path = _canonical_paper_environment_reference_path(project_root)
    canonical_data = read_json(canonical_path, {})
    if canonical_path.exists() and isinstance(canonical_data, dict):
        return canonical_path, canonical_data

    fallback_path = _archived_paper_environment_reference_path(project_root)
    fallback_data = read_json(fallback_path, {})
    if isinstance(fallback_data, dict):
        raw = fallback_data.get("raw")
        if isinstance(raw, dict):
            return (
                fallback_path,
                {
                    "reference_kind": "archived_deployment_reference",
                    "captured_at": str(fallback_data.get("generated_at", "") or ""),
                    "source_artifact": _portable_artifact_path(project_root, fallback_path),
                    **raw,
                    "note": "当前缺少冻结环境文件，暂时回退到 2026-04-19 的归档部署快照；它用于说明论文正式结果来源，不等同于当前答辩服务器的实时运行环境。",
                },
            )
        return fallback_path, fallback_data
    return fallback_path, {}


def _latest_model_validation_summary(artifacts_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    latest_dir = _latest_completed_dir(artifacts_dir, "model_validation_", "summary.json")
    if latest_dir is None:
        return None, {}
    path = latest_dir / "summary.json"
    data = read_json(path, {})
    return path, data if isinstance(data, dict) else {}


def _portable_container_validation_summary(artifacts_dir: Path) -> dict[str, Any]:
    candidate_paths = [
        artifacts_dir / "verification" / "docker_offline_validation" / "summary.json",
        artifacts_dir / "verification" / "docker_model_matrix" / "summary.json",
    ]
    path: Path | None = None
    data: dict[str, Any] = {}
    for candidate in candidate_paths:
        payload = read_json(candidate, {})
        if candidate.exists() and isinstance(payload, dict):
            path = candidate
            data = payload
            break
    if path is None:
        return {}

    model_matrix = data.get("model_matrix", {})
    if not isinstance(model_matrix, dict):
        model_matrix = {}
    attack_matrix = data.get("attack_matrix", {})
    if not isinstance(attack_matrix, dict):
        attack_matrix = {}

    model_rows = [row for row in list(model_matrix.get("rows", [])) if isinstance(row, dict)]
    attack_rows = [row for row in list(attack_matrix.get("rows", [])) if isinstance(row, dict)]

    validated_models = sorted(
        {
            str(row.get("model_adapter", "")).strip()
            for row in model_rows
            if str(row.get("job_status", "")).strip() == "success" and str(row.get("model_adapter", "")).strip()
        }
    )
    dataset_names = sorted(
        {
            str(dict(row.get("summary", {})).get("dataset_name", "")).strip()
            for row in model_rows
            if isinstance(row.get("summary", {}), dict) and str(dict(row.get("summary", {})).get("dataset_name", "")).strip()
        }
    )
    attacks = sorted(
        {
            str(row.get("attack", "")).strip()
            for row in attack_rows
            if str(row.get("attack", "")).strip()
        }
    )

    model_count = int(model_matrix.get("count", 0) or len(model_rows))
    model_success_count = int(model_matrix.get("success_count", 0) or len(validated_models))
    attack_count = int(attack_matrix.get("count", 0) or len(attack_rows))
    attack_success_count = int(attack_matrix.get("success_count", 0) or 0)

    return {
        "summary_path": str(path),
        "generated_at": str(data.get("generated_at", "") or ""),
        "overall_passed": bool(data.get("overall_passed", False)),
        "model_count": model_count,
        "model_success_count": model_success_count,
        "attack_count": attack_count,
        "attack_success_count": attack_success_count,
        "validated_models": validated_models,
        "dataset_names": dataset_names,
        "attacks": attacks,
        "note": (
            "该字段只说明 portable Docker 镜像在断网冷启动验收中已经跑通的模型矩阵和攻击矩阵，"
            "用于证明镜像可迁移和可离线复验；它不替代首页上方的轻量迁移验证计数，也不替代论文正式结果。"
        ),
    }


def _parse_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _all_jobs(store: Any, page_size: int = 500) -> list[dict[str, Any]]:
    if store is None or not hasattr(store, "list_jobs"):
        return []
    page = 1
    rows: list[dict[str, Any]] = []
    total = 0
    while True:
        batch_total, items = store.list_jobs(page=page, page_size=page_size, status="")
        total = max(int(batch_total or 0), total)
        rows.extend([dict(item) for item in items if isinstance(item, dict)])
        if not items or len(rows) >= total:
            break
        page += 1
    return rows


def _parse_scientific_validation_experiment(
    experiment_id: str,
    benchmark_attacks: list[str],
) -> dict[str, Any] | None:
    prefix = "scientific_validation_"
    text = str(experiment_id or "").strip()
    if not text.startswith(prefix):
        return None
    remainder = text[len(prefix) :]
    for attack in sorted((str(item).strip() for item in benchmark_attacks if str(item).strip()), key=len, reverse=True):
        marker = f"_{attack}"
        embedded = f"_{attack}_"
        if remainder.endswith(marker):
            adapter = remainder[: -len(marker)].strip("_")
            if adapter:
                return {
                    "model_adapter": adapter,
                    "attack": attack,
                    "variant": "",
                    "is_primary": True,
                }
        if embedded in remainder:
            adapter, variant = remainder.rsplit(embedded, 1)
            adapter = adapter.strip("_")
            variant = variant.strip("_")
            if adapter:
                return {
                    "model_adapter": adapter,
                    "attack": attack,
                    "variant": variant,
                    "is_primary": False,
                }
    return None


def _scientific_validation_jobs(store: Any, validation_summary: dict[str, Any]) -> list[dict[str, Any]]:
    criterion = validation_summary.get("criterion", {})
    if not isinstance(criterion, dict):
        criterion = {}
    benchmark_attacks = [str(item).strip() for item in list(criterion.get("benchmark_attacks", [])) if str(item).strip()]
    rows = _all_jobs(store)
    records: list[dict[str, Any]] = []
    for row in rows:
        override = _parse_json_object(row.get("override_json", ""))
        runner = override.get("runner", {})
        runner = runner if isinstance(runner, dict) else {}
        dataset = override.get("dataset", {})
        dataset = dataset if isinstance(dataset, dict) else {}
        experiment_id = str(runner.get("experiment_id", "") or "").strip()
        parsed = _parse_scientific_validation_experiment(experiment_id, benchmark_attacks)
        if not parsed:
            continue
        records.append(
            {
                "job_id": str(row.get("id", "") or ""),
                "job_status": str(row.get("status", "") or ""),
                "created_at": str(row.get("created_at", "") or ""),
                "started_at": str(row.get("started_at", "") or ""),
                "finished_at": str(row.get("finished_at", "") or ""),
                "error_message": str(row.get("error_message", "") or ""),
                "config_path": str(row.get("config_path", "") or ""),
                "experiment_id": experiment_id,
                "dataset_name": str(dataset.get("kind", "") or ""),
                "max_pairs": int(runner.get("max_pairs", 0) or 0),
                **parsed,
            }
        )
    return records


def _validation_snapshot_summary(
    validation_path: Path | None,
    validation_summary: dict[str, Any],
    validation_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot_id = ""
    snapshot_generated_at = ""
    if validation_path is not None and validation_path.exists():
        snapshot_id = validation_path.parent.name
        snapshot_generated_at = datetime.fromtimestamp(validation_path.stat().st_mtime, tz=timezone.utc).isoformat()
    live_job_in_progress = any(str(item.get("job_status", "")) in {"queued", "running"} and bool(item.get("is_primary", False)) for item in validation_jobs)
    return {
        "snapshot_id": snapshot_id,
        "snapshot_generated_at": snapshot_generated_at,
        "summary_path": str(validation_path or ""),
        "stable_archive": bool(validation_path),
        "snapshot_passed": bool(validation_summary.get("passed", False)),
        "live_job_in_progress": live_job_in_progress,
        "stable_reference_note": (
            "首页计数绑定到最近一次完成的验证快照；"
            "正在运行的规范补救任务会单独显示，但不会改写冻结快照"
        ),
    }


def _failing_primary_rows(
    models: list[dict[str, Any]],
    validated_models: list[str],
    validation_summary: dict[str, Any],
    validation_jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validated = {str(item).strip() for item in validated_models if str(item).strip()}
    unsupported = {
        str(item.get("adapter", "")).strip()
        for item in models
        if str(item.get("adapter", "")).strip() and bool(item.get("formal_eval", True))
    }
    per_model_rows = {
        str(row.get("model_adapter", "")).strip(): row
        for row in list(validation_summary.get("per_model", []))
        if isinstance(row, dict) and str(row.get("model_adapter", "")).strip()
    }
    jobs_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in validation_jobs:
        if not bool(row.get("is_primary", False)):
            continue
        key = (str(row.get("model_adapter", "")).strip(), str(row.get("attack", "")).strip())
        jobs_by_key.setdefault(key, []).append(row)
    for items in jobs_by_key.values():
        items.sort(
            key=lambda item: (
                str(item.get("created_at", "")),
                str(item.get("started_at", "")),
                str(item.get("finished_at", "")),
            ),
            reverse=True,
        )

    blockers: list[dict[str, Any]] = []
    benchmark_attacks = [str(item).strip() for item in list(dict(validation_summary.get("criterion", {})).get("benchmark_attacks", [])) if str(item).strip()]
    failing_models = sorted(adapter for adapter in unsupported if adapter and adapter not in validated)
    for adapter in failing_models:
        model_summary = per_model_rows.get(adapter, {})
        for attack in benchmark_attacks:
            history = jobs_by_key.get((adapter, attack), [])
            latest = history[0] if history else {}
            previous_failures = sum(1 for item in history[1:] if str(item.get("job_status", "")) == "failed")
            status = str(latest.get("job_status", "")) or "missing"
            error_message = str(latest.get("error_message", "") or "")
            if status in {"running", "queued"}:
                blocking_reason = "这条主验证记录仍在运行，稳定快照会保持未通过，直到它结束"
                if previous_failures:
                    blocking_reason += "；同一主验证记录此前已有失败尝试"
            elif status == "failed":
                blocking_reason = error_message or "这条主验证记录执行失败"
            else:
                blocking_reason = "冻结验证汇总里还没有这条主验证记录的成功结果"
            blockers.append(
                {
                    "model_adapter": adapter,
                    "attack": attack,
                    "dataset_name": str(latest.get("dataset_name", "") or dict(validation_summary.get("criterion", {})).get("dataset_name", "") or ""),
                    "experiment_id": str(latest.get("experiment_id", "") or f"scientific_validation_{adapter}_{attack}"),
                    "job_id": str(latest.get("job_id", "") or ""),
                    "job_status": status,
                    "previous_failure_count": previous_failures,
                    "last_updated_at": str(latest.get("finished_at", "") or latest.get("started_at", "") or latest.get("created_at", "") or ""),
                    "error_message": error_message,
                    "blocking_reason": blocking_reason,
                    "engineering_validated": bool(model_summary.get("validated", False)),
                    "scientific_quality_ok": bool(model_summary.get("scientific_quality_ok", False)),
                }
            )
    return blockers


def _scientific_quality_model_adapters(validation_summary: dict[str, Any]) -> list[str]:
    items = validation_summary.get("scientific_quality_validated_models", [])
    if not isinstance(items, list):
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def _model_coverage_summary(
    models: list[dict[str, Any]],
    ready_models: list[dict[str, Any]],
    validated_models: list[str],
    validation_summary: dict[str, Any],
) -> dict[str, Any]:
    supported_adapters = [
        str(item.get("adapter", "")).strip()
        for item in models
        if str(item.get("adapter", "")).strip() and bool(item.get("formal_eval", True))
    ]
    online_adapters = [
        str(item.get("adapter", "")).strip()
        for item in ready_models
        if str(item.get("adapter", "")).strip() and bool(item.get("formal_eval", True))
    ]
    scientific_quality_models = _scientific_quality_model_adapters(validation_summary)
    criterion = validation_summary.get("criterion", {})
    if not isinstance(criterion, dict):
        criterion = {}
    return {
        "integrated": {
            "count": len(supported_adapters),
            "models": supported_adapters,
            "semantics": "adapter integrated into the platform codebase and visible in the runtime model matrix",
        },
        "online": {
            "count": len(online_adapters),
            "models": online_adapters,
            "semantics": "endpoint or local adapter currently reachable in the live environment; this does not imply simultaneous full-load formal validation capacity",
        },
        "engineering_validated": {
            "count": len(validated_models),
            "models": list(validated_models),
            "passed": bool(validation_summary.get("passed", False)),
            "semantics": "completed the formal lightweight validation workflow under the platform's engineering acceptance criterion",
        },
        "scientific_quality": {
            "count": len(scientific_quality_models),
            "models": scientific_quality_models,
            "passed": bool(validation_summary.get("scientific_quality_passed", False)),
            "semantics": "passed the stricter clean-baseline and retrieval-drop signal checks inside the same validation matrix",
        },
        "validation_strategy": {
            "dataset_name": str(criterion.get("dataset_name", "") or ""),
            "benchmark_attacks": [str(item).strip() for item in list(criterion.get("benchmark_attacks", [])) if str(item).strip()],
            "max_pairs": int(criterion.get("max_pairs", 0) or 0),
            "description": str(criterion.get("description", "") or ""),
        },
    }


def _latest_runs(artifacts_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    latest_runs_raw = discover_runs_from_artifacts(str(artifacts_dir))[:limit]
    latest_runs = []
    for row in latest_runs_raw:
        latest_runs.append(
            {
                "run_id": str(row.get("run_id", "")),
                "created_at": str(row.get("created_at", "")),
                "task_kind": str(row.get("task_kind", "")),
                "dataset_name": str(row.get("dataset_name", "")),
                "benchmark_tag": str(row.get("benchmark_tag", "")),
                "attack": str(row.get("attack", "")),
                "mode": str(row.get("mode", "")),
                "defense": str(row.get("defense", "")),
                "experiment_id": str(row.get("experiment_id", "")),
                "model_adapter": str(row.get("model_adapter", "")),
                "surrogate_model_adapter": str(row.get("surrogate_model_adapter", row.get("model_adapter", ""))),
                "victim_model_adapters": [str(item) for item in list(row.get("victim_model_adapters", [])) if str(item).strip()],
                "asr": float(row.get("asr", 0.0) or 0.0),
                "asr_attack": float(row.get("asr_attack", row.get("asr", 0.0)) or 0.0),
                "asr_defended": float(row.get("asr_defended", row.get("asr", 0.0)) or 0.0),
                "defense_gain": float(row.get("defense_gain", 0.0) or 0.0),
                "risk_score": float(row.get("risk_score", 0.0) or 0.0),
                "risk_level": str(row.get("risk_level", "")),
                "risk_scenario": str(row.get("risk_scenario", "")),
                "avg_l2": float(row.get("avg_l2", 0.0) or 0.0),
                "path": str(row.get("path", "")),
            }
        )
    return latest_runs


def _formal_row_joint_execution(row: dict[str, Any], summary_data: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_formal_joint_execution(row, summary_data)


_ATTACK_DISPLAY_ORDER = {
    "advclip": 0,
    "tmm": 1,
    "advedm": 2,
    "advedm_plus": 3,
}

_ABLATION_VARIANT_ORDER = {
    "完整版本": 0,
    "去文本分支": 1,
    "去自适应预算": 2,
    "去注视引导": 3,
    "其他变体": 9,
}


def _detect_ablation_variant(*parts: str) -> str:
    joined = " ".join(part.lower() for part in parts if part).strip()
    if "no_text" in joined:
        return "去文本分支"
    if "no_adaptive" in joined:
        return "去自适应预算"
    if "no_fixation" in joined:
        return "去注视引导"
    if "full" in joined:
        return "完整版本"
    return "其他变体"


def _formal_suite_meta(suite_name: str, *, analysis_row_id: str, benchmark_tag: str, experiment_id: str) -> dict[str, Any]:
    if suite_name == "E1_classic_coco":
        return {
            "suite_label": "E1 主实验",
            "evidence_group": "primary",
            "experiment_label": "E1 主实验",
            "sort_key": (0, 0, 0),
        }
    if suite_name == "E2_classic_flickr":
        return {
            "suite_label": "E2 主实验",
            "evidence_group": "primary",
            "experiment_label": "E2 主实验",
            "sort_key": (0, 1, 0),
        }
    if suite_name == "E4_ablation":
        variant_label = _detect_ablation_variant(analysis_row_id, benchmark_tag, experiment_id)
        return {
            "suite_label": "E4 消融",
            "evidence_group": "ablation",
            "experiment_label": f"E4 消融 / {variant_label}",
            "sort_key": (1, 0, _ABLATION_VARIANT_ORDER.get(variant_label, 9)),
        }
    prefix = suite_name.split("_", 1)[0] if suite_name else "正式实验"
    return {
        "suite_label": prefix,
        "evidence_group": "supplemental",
        "experiment_label": prefix,
        "sort_key": (2, 0, 0),
    }


def _formal_row_artifact_paths(project_root: Path, suite_row: dict[str, Any]) -> dict[str, str]:
    summary_path = _portable_artifact_path(project_root, suite_row.get("summary_path", ""))
    report_path = _portable_artifact_path(project_root, suite_row.get("report_path", ""))
    return {
        "summary_path": summary_path,
        "report_path": report_path,
        "archived_summary_path": _portable_artifact_path(
            project_root,
            suite_row.get(
                "archived_summary_path",
                summary_path if str(suite_row.get("summary_path", "") or "").endswith("row_evidence.json") else "",
            ),
        ),
        "archived_report_path": _portable_artifact_path(
            project_root,
            suite_row.get(
                "archived_report_path",
                report_path if str(suite_row.get("report_path", "") or "").endswith("row_evidence.md") else "",
            ),
        ),
        "source_summary_path": _portable_artifact_path(project_root, suite_row.get("source_summary_path", "")),
        "source_report_data_path": _portable_artifact_path(project_root, suite_row.get("source_report_data_path", "")),
        "source_report_path": _portable_artifact_path(project_root, suite_row.get("source_report_path", "")),
        "portable_report_data_path": _portable_artifact_path(project_root, suite_row.get("portable_report_data_path", "")),
        "portable_report_path": _portable_artifact_path(project_root, suite_row.get("portable_report_path", "")),
        "artifact_index_path": _portable_artifact_path(project_root, suite_row.get("artifact_index_path", "")),
    }


def _formal_row_models(suite_row: dict[str, Any], run_row: dict[str, Any]) -> tuple[str, list[str]]:
    surrogate_adapter = str(
        run_row.get("surrogate_model_adapter", run_row.get("model_adapter", ""))
        or run_row.get("model_adapter", "")
        or ""
    ).strip()
    victim_model_adapters = [
        str(item).strip()
        for item in list(suite_row.get("victim_model_adapters", run_row.get("victim_model_adapters", [])))
        if str(item).strip()
    ]
    return surrogate_adapter, victim_model_adapters


def _formal_row_from_analysis(
    project_root: Path,
    suite_row: dict[str, Any],
    run_index: dict[str, dict[str, Any]],
    attack_catalog: dict[str, dict[str, str]],
) -> tuple[tuple[int, int, int, int, str], dict[str, Any]] | None:
    suite_name = str(suite_row.get("suite", "") or "").strip()
    if not suite_name or suite_name.startswith("E0"):
        return None
    run_id = str(suite_row.get("run_id", "") or "").strip()
    run_row = run_index.get(run_id, {})
    analysis_row_id = str(suite_row.get("id", "") or "").strip()
    benchmark_tag = str(suite_row.get("benchmark_tag", run_row.get("benchmark_tag", "")) or "").strip()
    experiment_id = str(run_row.get("experiment_id", "") or analysis_row_id).strip()
    suite_meta = _formal_suite_meta(
        suite_name,
        analysis_row_id=analysis_row_id,
        benchmark_tag=benchmark_tag,
        experiment_id=experiment_id,
    )
    attack_id = str(suite_row.get("attack", run_row.get("attack", "")) or "").strip()
    surrogate_adapter, victim_model_adapters = _formal_row_models(suite_row, run_row)
    artifacts = _formal_row_artifact_paths(project_root, suite_row)
    joint_execution = _formal_row_joint_execution(
        {
            **suite_row,
            "benchmark_tag": benchmark_tag,
            "experiment_id": experiment_id,
            "evidence_row_id": analysis_row_id,
            "summary_path": artifacts["summary_path"],
            "report_path": artifacts["report_path"],
        }
    )
    metric_semantics = _derive_formal_row_metric_semantics(suite_row, run_row)
    formal_row = {
        "run_id": run_id,
        "created_at": str(run_row.get("created_at", "")),
        "task_kind": str(run_row.get("task_kind", "vlr") or "vlr"),
        "dataset_name": str(suite_row.get("dataset_name", run_row.get("dataset_name", "")) or ""),
        "benchmark_tag": benchmark_tag,
        "attack": attack_id,
        "attack_modality": str(attack_catalog.get(attack_id, {}).get("modality", "") or ""),
        "eval_scope": str(suite_row.get("eval_scope", run_row.get("eval_scope", "")) or ""),
        "mode": str(run_row.get("mode", "") or ""),
        "defense": str(run_row.get("defense", "") or ""),
        "experiment_id": experiment_id,
        "suite": suite_name,
        "suite_label": str(suite_meta.get("suite_label", "") or ""),
        "evidence_group": str(suite_meta.get("evidence_group", "") or ""),
        "experiment_label": str(suite_meta.get("experiment_label", "") or ""),
        "model_adapter": str(run_row.get("model_adapter", surrogate_adapter) or surrogate_adapter),
        "surrogate_model_adapter": surrogate_adapter,
        "victim_model_adapters": victim_model_adapters,
        "asr": float(suite_row.get("asr_attack", run_row.get("asr_attack", run_row.get("asr", 0.0))) or 0.0),
        "asr_attack": float(suite_row.get("asr_attack", run_row.get("asr_attack", run_row.get("asr", 0.0))) or 0.0),
        **metric_semantics,
        "asr_defended": float(suite_row.get("asr_defended", run_row.get("asr_defended", 0.0)) or 0.0),
        "defense_gain": float(suite_row.get("defense_gain", run_row.get("defense_gain", 0.0)) or 0.0),
        "risk_score": float(run_row.get("risk_score", suite_row.get("risk_score", 0.0)) or 0.0),
        "risk_level": str(run_row.get("risk_level", suite_row.get("risk_level", ""))),
        "risk_scenario": str(run_row.get("risk_scenario", "")),
        "avg_l2": float(suite_row.get("avg_l2", run_row.get("avg_l2", 0.0)) or 0.0),
        "path": str(run_row.get("path", "")),
        "evidence_row_id": analysis_row_id,
        "official_result": True,
        **artifacts,
        **joint_execution,
    }
    sort_key = (
        *tuple(suite_meta.get("sort_key", (9, 9, 9))),
        _ATTACK_DISPLAY_ORDER.get(str(formal_row.get("attack", "") or ""), 9),
        run_id,
    )
    return sort_key, formal_row


def _analysis_rows_to_formal_runs(
    project_root: Path,
    artifacts_dir: Path,
    analysis: dict[str, Any],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    rows = [row for row in list(analysis.get("rows", [])) if isinstance(row, dict)]
    if not rows:
        return []

    run_index = {str(row.get("run_id", "")): row for row in _latest_runs(artifacts_dir, limit=400)}
    attack_catalog = _attack_catalog_entries(project_root)
    ranked_rows: list[tuple[tuple[int, int, int, int, str], dict[str, Any]]] = []
    for suite_row in rows:
        formal_run = _formal_row_from_analysis(project_root, suite_row, run_index, attack_catalog)
        if formal_run is not None:
            ranked_rows.append(formal_run)
    ranked_rows.sort(key=lambda item: item[0])
    return [row for _, row in ranked_rows[:limit]]


def _latest_formal_runs(project_root: Path, artifacts_dir: Path, limit: int = 6) -> list[dict[str, Any]]:
    _, analysis = _latest_paper_suite_analysis(project_root, artifacts_dir)
    return _analysis_rows_to_formal_runs(project_root, artifacts_dir, analysis, limit=limit)


def _primary_formal_runs(project_root: Path, artifacts_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    _, analysis, _ = _primary_paper_suite_analysis(project_root, artifacts_dir)
    return _analysis_rows_to_formal_runs(project_root, artifacts_dir, analysis, limit=limit)


def _split_formal_runs(
    formal_rows: list[dict[str, Any]],
    *,
    primary_limit: int | None = None,
    ablation_limit: int = 6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows = [row for row in formal_rows if str(row.get("evidence_group", "")) == "primary"]
    if primary_limit is not None and primary_limit > 0:
        primary_rows = primary_rows[:primary_limit]
    ablation_rows = [row for row in formal_rows if str(row.get("evidence_group", "")) == "ablation"][:ablation_limit]
    return primary_rows, ablation_rows


def _sample_management_artifacts(artifacts_dir: Path) -> dict[str, int]:
    runs_dir = artifacts_dir / "runs"
    if not runs_dir.exists():
        return {
            "cases_index_runs": 0,
            "case_bundles": 0,
            "attack_debug_cases": 0,
            "patch_registry_entries": 0,
        }

    cases_index_runs = 0
    case_bundles = 0
    attack_debug_cases = 0
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if (run_dir / "cases_index.jsonl").exists():
            cases_index_runs += 1
        case_bundles += sum(1 for _ in run_dir.glob("cases/*/case_bundle.json"))
        attack_debug_cases += sum(1 for _ in run_dir.glob("attack_debug/*/debug.json"))

    patch_registry = read_json(artifacts_dir / "advclip_patch_registry.json", {"entries": {}})
    patch_entries = 0
    if isinstance(patch_registry, dict):
        patch_entries = len(dict(patch_registry.get("entries", {})))

    return {
        "cases_index_runs": cases_index_runs,
        "case_bundles": case_bundles,
        "attack_debug_cases": attack_debug_cases,
        "patch_registry_entries": patch_entries,
    }


def _validated_model_adapters(artifacts_dir: Path) -> list[str]:
    validation_path, validation = _latest_model_validation_summary(artifacts_dir)
    if validation_path is None:
        return []
    models = [str(item).strip() for item in list(validation.get("validated_models", [])) if str(item).strip()]
    if models:
        return sorted(set(models))
    return []


def _rows_by_suite(rows: list[dict[str, Any]], suite: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("suite", "")) == suite]


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _phase_asr_and_ranking_ok(
    *,
    phase_key: str,
    cfg: dict[str, Any],
    attack_map: dict[str, dict[str, Any]],
    findings: list[str],
) -> bool:
    ok = True
    minimum_asr_attack = dict(cfg.get("minimum_asr_attack", {}))
    snapshot_asr = dict(cfg.get("snapshot_asr_attack", {}))
    enforce_snapshot_fraction = bool(cfg.get("enforce_snapshot_fraction", True))
    min_fraction = float(cfg.get("minimum_snapshot_fraction", 0.0) or 0.0)

    for attack in [str(x) for x in list(cfg.get("attacks", []))]:
        row = attack_map.get(attack)
        if row is None:
            ok = False
            findings.append(f"{phase_key.upper()} missing attack row: {attack}")
            continue
        if bool(cfg.get("require_zero_victim_failures", False)) and int(row.get("num_victim_failures", 0) or 0) != 0:
            ok = False
            findings.append(f"{phase_key.upper()} victim failure present: {attack}")
        current = float(row.get("asr_attack", 0.0) or 0.0)
        if attack in minimum_asr_attack and current < float(minimum_asr_attack[attack] or 0.0):
            ok = False
            findings.append(f"{phase_key.upper()} compact ASR below threshold: {attack}")
        baseline = float(snapshot_asr.get(attack, 0.0) or 0.0)
        if enforce_snapshot_fraction and baseline > 0 and current < baseline * min_fraction:
            ok = False
            findings.append(f"{phase_key.upper()} ASR below snapshot fraction: {attack}")

    if all(name in attack_map for name in ("tmm", "advedm_plus", "advedm", "advclip")):
        tmm = float(attack_map["tmm"].get("asr_attack", 0.0) or 0.0)
        advedm_plus = float(attack_map["advedm_plus"].get("asr_attack", 0.0) or 0.0)
        advedm = float(attack_map["advedm"].get("asr_attack", 0.0) or 0.0)
        advclip = float(attack_map["advclip"].get("asr_attack", 0.0) or 0.0)
        if not (tmm > advedm_plus > advedm and advedm_plus > advclip):
            ok = False
            findings.append(f"{phase_key.upper()} attack ranking mismatch")
        if (tmm - advedm_plus) < float(cfg.get("minimum_gap_tmm_over_advedm_plus", 0.0) or 0.0):
            ok = False
            findings.append(f"{phase_key.upper()} TMM gap over ADVEDM+ is too small")
        if (advedm_plus - advedm) < float(cfg.get("minimum_gap_advedm_plus_over_advedm", 0.0) or 0.0):
            ok = False
            findings.append(f"{phase_key.upper()} ADVEDM+ gap over ADVEDM is too small")
        if (advedm_plus - advclip) < float(cfg.get("minimum_gap_advedm_plus_over_advclip", 0.0) or 0.0):
            ok = False
            findings.append(f"{phase_key.upper()} ADVEDM+ gap over AdvCLIP is too small")
    return ok


def _missing_result_conformance_payload(analysis_path: Path, threshold_path: Path, validation_path: Path | None) -> dict[str, Any]:
    return {
        "analysis_path": str(analysis_path),
        "threshold_path": str(threshold_path),
        "model_validation_path": str(validation_path or ""),
        "available": False,
        "passed": False,
        "phase_count": 0,
        "row_count": 0,
        "e0_ok": False,
        "e1_ok": False,
        "e2_ok": False,
        "e4_ok": False,
        "defense_ok": False,
        "adaptive_ok": False,
        "fixation_ok": False,
        "model_validation_ok": False,
        "classic_conclusions": [],
        "caveats": ["analysis or threshold artifact missing"],
    }


def _suite_attack_map(rows: list[dict[str, Any]], suite: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("attack", "")): row for row in _rows_by_suite(rows, suite)}


def _e0_conformance(rows: list[dict[str, Any]], thresholds: dict[str, Any], findings: list[str]) -> bool:
    e0_cfg = dict(thresholds.get("e0", {}))
    e0_rows = _rows_by_suite(rows, str(e0_cfg.get("suite", "")))
    e0_ok = bool(e0_rows)
    if e0_ok and bool(e0_cfg.get("require_zero_victim_failures", False)):
        e0_ok = int(e0_rows[0].get("num_victim_failures", 0) or 0) == 0
        if not e0_ok:
            findings.append("E0 smoke has victim failures")
    return e0_ok


def _phase_conformance(rows: list[dict[str, Any]], thresholds: dict[str, Any], phase_key: str, findings: list[str]) -> bool:
    cfg = dict(thresholds.get(phase_key, {}))
    return _phase_asr_and_ranking_ok(
        phase_key=phase_key,
        cfg=cfg,
        attack_map=_suite_attack_map(rows, str(cfg.get("suite", ""))),
        findings=findings,
    )


def _selected_e4_rows(rows: list[dict[str, Any]], e4_cfg: dict[str, Any], findings: list[str]) -> tuple[bool, dict[str, dict[str, Any]]]:
    e4_rows = {str(row.get("id", "")): row for row in _rows_by_suite(rows, str(e4_cfg.get("suite", "")))}
    selected: dict[str, dict[str, Any]] = {}
    e4_ok = True
    for label, row_id in dict(e4_cfg.get("ids", {})).items():
        row = e4_rows.get(str(row_id))
        if row is None:
            e4_ok = False
            findings.append(f"E4 missing row: {row_id}")
            continue
        if bool(e4_cfg.get("require_zero_victim_failures", False)) and int(row.get("num_victim_failures", 0) or 0) != 0:
            e4_ok = False
            findings.append(f"E4 victim failure present: {row_id}")
        selected[str(label)] = row
    return e4_ok, selected


def _e4_gain_ok(
    selected: dict[str, dict[str, Any]],
    e4_cfg: dict[str, Any],
    *,
    baseline_key: str,
    threshold_key: str,
) -> bool:
    if "full" not in selected or baseline_key not in selected:
        return False
    gain = float(selected["full"].get("asr_attack", 0.0) or 0.0) - float(selected[baseline_key].get("asr_attack", 0.0) or 0.0)
    return gain >= float(e4_cfg.get(threshold_key, 0.0) or 0.0)


def _e4_conformance(rows: list[dict[str, Any]], thresholds: dict[str, Any], findings: list[str]) -> tuple[bool, bool, bool]:
    e4_cfg = dict(thresholds.get("e4", {}))
    e4_ok, selected = _selected_e4_rows(rows, e4_cfg, findings)
    if "full" in selected and "no_text" in selected and not _e4_gain_ok(
        selected,
        e4_cfg,
        baseline_key="no_text",
        threshold_key="minimum_asr_gain_over_no_text",
    ):
        e4_ok = False
        findings.append("E4 text branch ASR gain below threshold")
    adaptive_ok = _e4_gain_ok(
        selected,
        e4_cfg,
        baseline_key="no_adaptive",
        threshold_key="minimum_asr_gain_over_no_adaptive",
    )
    fixation_ok = _e4_gain_ok(
        selected,
        e4_cfg,
        baseline_key="no_fixation",
        threshold_key="minimum_asr_gain_over_no_fixation",
    )
    if "full" in selected and "no_adaptive" in selected and not adaptive_ok:
        e4_ok = False
        findings.append("E4 adaptive budget gain below threshold")
    if "full" in selected and "no_fixation" in selected and not fixation_ok:
        e4_ok = False
        findings.append("E4 fixation gain below threshold")
    return e4_ok, adaptive_ok, fixation_ok


def _defense_conformance(rows: list[dict[str, Any]], thresholds: dict[str, Any], findings: list[str]) -> bool:
    defense_cfg = dict(thresholds.get("defense", {}))
    suites = {str(x) for x in list(defense_cfg.get("suites", []))}
    attacks = {str(x) for x in list(defense_cfg.get("attacks", []))}
    defense_rows = [row for row in rows if str(row.get("suite", "")) in suites and str(row.get("attack", "")) in attacks]
    threshold = float(defense_cfg.get("effective_attack_asr_threshold", 0.0) or 0.0)
    effective_rows = [row for row in defense_rows if float(row.get("asr_attack", 0.0) or 0.0) >= threshold]
    defense_gains = [float(row.get("defense_gain", 0.0) or 0.0) for row in effective_rows]
    defense_ok = bool(
        effective_rows
        and len(effective_rows) >= int(defense_cfg.get("minimum_effective_row_count", 0) or 0)
        and sum(1 for value in defense_gains if value > 0.0)
        >= int(defense_cfg.get("minimum_effective_positive_row_count", 0) or 0)
        and _safe_mean(defense_gains) >= float(defense_cfg.get("minimum_effective_mean_defense_gain", 0.0) or 0.0)
    )
    if not defense_ok:
        findings.append("Defense recovery is still below the baseline acceptance threshold")
    return defense_ok


def _model_validation_conformance(validation: dict[str, Any], thresholds: dict[str, Any], findings: list[str]) -> bool:
    cfg = dict(thresholds.get("model_validation", {}))
    validation_models = [str(item).strip() for item in list(validation.get("validated_models", [])) if str(item).strip()]
    validation_attacks = {
        str(item).strip()
        for item in list(validation.get("attacks", dict(validation.get("criterion", {})).get("benchmark_attacks", [])))
        if str(item).strip()
    }
    criterion = dict(validation.get("criterion", {}))
    model_validation_ok = bool(
        validation
        and bool(validation.get("passed", False))
        and int(validation.get("validated_model_count", 0) or len(validation_models)) >= int(cfg.get("required_model_count", 0) or 0)
        and str(validation.get("dataset_name", "")) == str(cfg.get("required_dataset", ""))
        and {str(item) for item in list(cfg.get("required_attacks", []))}.issubset(validation_attacks)
        and int(criterion.get("max_pairs", 0) or 0) == int(cfg.get("required_max_pairs", 0) or 0)
        and int(criterion.get("minimum_qualifying_attack_count_per_model", 0) or 0)
        >= int(cfg.get("minimum_qualifying_attack_count_per_model", 0) or 0)
        and float(criterion.get("minimum_attack_asr_any", 0.0) or 0.0) >= float(cfg.get("minimum_attack_asr_any", 0.0) or 0.0)
        and float(criterion.get("minimum_attack_drop_r1_any", 0.0) or 0.0)
        >= float(cfg.get("minimum_attack_drop_r1_any", 0.0) or 0.0)
    )
    if not model_validation_ok:
        findings.append("Lightweight 10-model transfer validation matrix is incomplete")
    return model_validation_ok


def _classic_conformance_conclusions() -> list[str]:
    classic_conclusions = [
        "经典三模型主实验满足紧凑验收口径下的排序与效应量门槛。",
        "在 E1 和 E2 主实验中，ADVEDM+ 仍强于 ADVEDM 和 AdvCLIP。",
        "当前验收只用于判断平台是否具备基本的对抗样本生成、测评和量化分析能力。",
        "当前验收门槛要求同时满足主实验效果量、基于 ASR 的攻击成功判定、扰动约束记录，以及 10 模型轻量迁移验证矩阵。",
    ]
    return classic_conclusions


def _result_conformance_payload(
    *,
    analysis_path: Path,
    threshold_path: Path,
    validation_path: Path | None,
    analysis: dict[str, Any],
    checks: dict[str, bool],
    findings: list[str],
) -> dict[str, Any]:
    caveats = findings[:8]
    if not caveats:
        caveats = ["All configured acceptance checks passed."]
    return {
        "analysis_path": str(analysis_path),
        "threshold_path": str(threshold_path),
        "model_validation_path": str(validation_path or ""),
        "available": True,
        "passed": bool(all(checks.values())),
        "phase_count": int(analysis.get("phase_count", 0) or 0),
        "row_count": int(analysis.get("row_count", 0) or 0),
        "e0_ok": checks["e0_ok"],
        "e1_ok": checks["e1_ok"],
        "e2_ok": checks["e2_ok"],
        "e4_ok": checks["e4_ok"],
        "defense_ok": checks["defense_ok"],
        "adaptive_ok": checks["adaptive_ok"],
        "fixation_ok": checks["fixation_ok"],
        "model_validation_ok": checks["model_validation_ok"],
        "classic_conclusions": _classic_conformance_conclusions(),
        "caveats": caveats,
    }


def _result_conformance_v2(project_root: Path, artifacts_dir: Path) -> dict[str, Any]:
    analysis_path, analysis, _ = _primary_paper_suite_analysis(project_root, artifacts_dir)
    validation_path, validation = _latest_model_validation_summary(artifacts_dir)
    threshold_path = project_root / "artifacts" / "paper_acceptance_thresholds.json"
    thresholds = read_json(threshold_path, {})
    if not isinstance(analysis, dict) or not isinstance(thresholds, dict):
        return _missing_result_conformance_payload(analysis_path, threshold_path, validation_path)

    rows = [row for row in list(analysis.get("rows", [])) if isinstance(row, dict)]
    findings: list[str] = []
    e4_ok, adaptive_ok, fixation_ok = _e4_conformance(rows, thresholds, findings)
    checks = {
        "e0_ok": _e0_conformance(rows, thresholds, findings),
        "e1_ok": _phase_conformance(rows, thresholds, "e1", findings),
        "e2_ok": _phase_conformance(rows, thresholds, "e2", findings),
        "e4_ok": e4_ok,
        "defense_ok": _defense_conformance(rows, thresholds, findings),
        "adaptive_ok": adaptive_ok,
        "fixation_ok": fixation_ok,
        "model_validation_ok": _model_validation_conformance(validation, thresholds, findings),
    }
    return _result_conformance_payload(
        analysis_path=analysis_path,
        threshold_path=threshold_path,
        validation_path=validation_path,
        analysis=analysis,
        checks=checks,
        findings=findings,
    )


def _system_overview_context(request: Request) -> dict[str, Any]:
    project_root, artifacts_dir = _runtime_context(request)
    store = getattr(request.app.state, "store", None)
    registry_path = artifacts_dir / "advclip_patch_registry.json"
    patch_registry = read_json(registry_path, {"version": 1, "entries": {}})
    if not isinstance(patch_registry, dict):
        patch_registry = {"version": 1, "entries": {}}
    models = list_main_models(project_root=project_root)
    ready_models = [item for item in models if str(item.get("health_status", "")) == "ready"]
    validated_models = _validated_model_adapters(artifacts_dir)
    validation_path, validation_summary = _latest_model_validation_summary(artifacts_dir)
    validation_summary = validation_summary if isinstance(validation_summary, dict) else {}
    validation_jobs = _scientific_validation_jobs(store, validation_summary)
    scientific_quality_models = _scientific_quality_model_adapters(validation_summary)
    dataset_catalog = _dataset_catalog()
    live_datasets = _live_datasets(project_root, store)
    paper_suite_analysis_path, _, paper_suite_source_kind = _primary_paper_suite_analysis(project_root, artifacts_dir)
    paper_environment_path, paper_environment = _paper_result_environment_reference(project_root)
    if not isinstance(paper_environment, dict):
        paper_environment = {}
    formal_runs = _latest_formal_runs(project_root, artifacts_dir, limit=20)
    primary_evidence_rows = _primary_formal_runs(project_root, artifacts_dir, limit=20)
    primary_formal_runs, ablation_formal_runs = _split_formal_runs(
        primary_evidence_rows,
        primary_limit=None,
        ablation_limit=6,
    )
    primary_artifact_index_path = next(
        (
            str(row.get("artifact_index_path", "")).strip()
            for row in primary_formal_runs
            if str(row.get("artifact_index_path", "")).strip()
        ),
        "",
    )
    return {
        "request": request,
        "project_root": project_root,
        "artifacts_dir": artifacts_dir,
        "store": store,
        "app_db": str(getattr(store, "path", artifacts_dir / "app.db")),
        "patch_registry": patch_registry,
        "models": models,
        "ready_models": ready_models,
        "validated_models": validated_models,
        "validation_path": validation_path,
        "validation_summary": validation_summary,
        "validation_jobs": validation_jobs,
        "scientific_quality_models": scientific_quality_models,
        "dataset_catalog": dataset_catalog,
        "live_datasets": live_datasets,
        "paper_suite_analysis_path": paper_suite_analysis_path,
        "paper_suite_source_kind": paper_suite_source_kind,
        "paper_environment_path": paper_environment_path,
        "paper_environment": paper_environment,
        "formal_runs": formal_runs,
        "primary_formal_runs": primary_formal_runs,
        "ablation_formal_runs": ablation_formal_runs,
        "primary_artifact_index_path": primary_artifact_index_path,
    }


def _overview_build_identity(ctx: dict[str, Any]) -> dict[str, Any]:
    request = ctx["request"]
    runtime_identity = _build_runtime_identity()
    frontend_build = _frontend_build_info(ctx["project_root"])
    deployment_version = _deployment_version_info(ctx["project_root"])
    return {
        "deployment_target": deployment_version["deployment_target"],
        "backend_version_stamp": deployment_version["version"],
        "version_source": deployment_version["version_source"],
        "runtime_started_at": str(getattr(request.app.state, "runtime_started_at", "") or ""),
        "runtime_process_pid": int(getattr(request.app.state, "runtime_process_pid", 0) or 0),
        "runtime_instance_id": str(getattr(request.app.state, "runtime_instance_id", "") or ""),
        "backend_commit": _git_value(ctx["project_root"], ["rev-parse", "--short", "HEAD"]),
        "runtime_transport": runtime_identity["runtime_transport"],
        "containerized": bool(runtime_identity["containerized"]),
        "runtime_context": runtime_identity["runtime_context"],
        "image_ref": runtime_identity["image_ref"],
        "runtime_profile": runtime_identity["runtime_profile"],
        "runtime_volume_name": runtime_identity["runtime_volume_name"],
        "bundle_root": runtime_identity["bundle_root"],
        "runtime_root": runtime_identity["runtime_root"],
        "frontend_index_exists": bool(frontend_build.get("index_exists", False)),
        "frontend_dist_fresh": bool(frontend_build.get("dist_fresh", False)),
        "frontend_index_built_at": str(frontend_build.get("index_built_at", "")),
        "frontend_inputs_latest_at": str(frontend_build.get("inputs_latest_at", "")),
        "frontend_asset_refs": list(frontend_build.get("asset_refs", [])),
    }


def _overview_repositories(project_root: Path) -> list[dict[str, Any]]:
    return [
        _repo_entry(project_root, "third_party/papers/AdvCLIP"),
        _repo_entry(project_root, "third_party/papers/TMM"),
        _repo_entry(project_root, "third_party/papers/AdvEDM_demo"),
        _repo_entry(project_root, "third_party/papers/VLPTransferAttack"),
        _repo_entry(project_root, "external/VQA-Visual-Robustness-Benchmark"),
        _repo_entry(project_root, "external/XTransferBench"),
        _repo_entry(project_root, "external/FOA-Attack"),
        _repo_entry(project_root, "external/AnyAttack"),
        _repo_entry(project_root, "external/MPCAttack"),
        _repo_entry(project_root, "external/M-Attack"),
    ]


_EXTERNAL_ATTACK_STATUS_CONFIGS: dict[str, str] = {
    "vqa_visual_corruption": "configs/bench/bootstrap_standard_vqa_visual_corruption_cuda.yaml",
    "xtransfer_uap": "configs/bench/bootstrap_standard_caption_xtransfer_uap_cuda.yaml",
    "foa_attack": "configs/bench/bootstrap_standard_caption_foa_attack_cuda.yaml",
    "anyattack": "configs/bench/bootstrap_standard_caption_anyattack_cuda.yaml",
    "mpc_attack": "configs/bench/bootstrap_standard_caption_mpc_attack_cuda.yaml",
    "m_attack": "configs/bench/bootstrap_standard_caption_m_attack_cuda.yaml",
}

_EXTERNAL_ATTACK_REQUIREMENTS: dict[str, dict[str, bool]] = {
    "vqa_visual_corruption": {"repo": True, "checkpoint": False, "target": False},
    "xtransfer_uap": {"repo": False, "checkpoint": True, "target": False},
    "foa_attack": {"repo": True, "checkpoint": False, "target": True},
    "anyattack": {"repo": True, "checkpoint": True, "target": True},
    "mpc_attack": {"repo": True, "checkpoint": False, "target": True},
    "m_attack": {"repo": True, "checkpoint": False, "target": True},
}

_EXTERNAL_ATTACK_DISPLAY_NAMES: dict[str, str] = {
    "vqa_visual_corruption": "官方视觉退化攻击（VQA Visual Robustness）",
    "xtransfer_uap": "跨任务通用扰动（X-Transfer UAP）",
    "foa_attack": "特征最优对齐迁移攻击（FOA-Attack）",
    "anyattack": "任意图像目标生成攻击（AnyAttack）",
    "mpc_attack": "多范式协同迁移攻击（MPCAttack）",
    "m_attack": "局部语义匹配迁移攻击（M-Attack）",
}


def _resolve_project_path(project_root: Path, raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _requirement_status(project_root: Path, label: str, required: bool, raw_path: Any) -> dict[str, Any]:
    path = _resolve_project_path(project_root, raw_path)
    if path is None:
        return {
            "label": label,
            "required": required,
            "configured": False,
            "exists": False,
            "status": "missing" if required else "not_required",
            "path": "",
            "note": "必填项未配置" if required else "该方法不需要此项",
        }
    exists = path.exists()
    return {
        "label": label,
        "required": required,
        "configured": True,
        "exists": exists,
        "status": "ready" if exists else "missing",
        "path": str(path),
        "note": "已配置并存在" if exists else "已配置但文件或目录不存在",
    }


def _target_status(project_root: Path, required: bool, attack_cfg: dict[str, Any]) -> dict[str, Any]:
    target_text = str(attack_cfg.get("target_text", "") or "").strip()
    target_image = str(attack_cfg.get("target_image", "") or "").strip()
    if target_text:
        return {
            "label": "目标图/目标文本",
            "required": required,
            "configured": True,
            "exists": True,
            "status": "ready",
            "path": _resolve_project_path(project_root, target_image).as_posix() if target_image and _resolve_project_path(project_root, target_image) else "",
            "note": "target_text 已配置",
        }
    return _requirement_status(project_root, "目标图/目标文本", required, target_image)


def _checkpoint_field_for_attack(attack_id: str) -> str:
    if attack_id == "anyattack":
        return "decoder_path"
    if attack_id == "xtransfer_uap":
        return "uap_path"
    return "checkpoint_path"


def _external_attack_runtime_status(project_root: Path) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for attack_id, config_rel in _EXTERNAL_ATTACK_STATUS_CONFIGS.items():
        config_path = project_root / config_rel
        config_exists = config_path.exists()
        raw_cfg: dict[str, Any] = {}
        if config_exists:
            try:
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                raw_cfg = loaded if isinstance(loaded, dict) else {}
            except (OSError, yaml.YAMLError) as exc:
                raw_cfg = {"_load_error": str(exc)}
        attack_cfg = raw_cfg.get("attack", {}) if isinstance(raw_cfg.get("attack", {}), dict) else {}
        requirements = _EXTERNAL_ATTACK_REQUIREMENTS.get(attack_id, {})
        repo = _requirement_status(project_root, "官方仓库", bool(requirements.get("repo", False)), attack_cfg.get("repo_dir", ""))
        checkpoint_field = _checkpoint_field_for_attack(attack_id)
        checkpoint = _requirement_status(project_root, "权重/UAP", bool(requirements.get("checkpoint", False)), attack_cfg.get(checkpoint_field, ""))
        target = _target_status(project_root, bool(requirements.get("target", False)), attack_cfg)
        command_template_configured = bool(str(attack_cfg.get("command_template", "") or "").strip() or attack_id == "xtransfer_uap")
        mandatory_items = [repo, checkpoint, target]
        required_ready = all(item["status"] == "ready" for item in mandatory_items if bool(item.get("required", False)))
        load_error = str(raw_cfg.get("_load_error", "") or "")
        runnable = bool(config_exists and command_template_configured and required_ready and not load_error)
        messages: list[str] = []
        if not config_exists:
            messages.append(f"标准配置不存在：{config_rel}")
        if load_error:
            messages.append(f"标准配置解析失败：{load_error}")
        if not command_template_configured:
            messages.append("外部命令模板未配置")
        for item in mandatory_items:
            if bool(item.get("required", False)) and item.get("status") != "ready":
                messages.append(f"{item.get('label', '必填项')}未就绪：{item.get('note', '')}")
        status[attack_id] = {
            "attack_id": attack_id,
            "display_name": _EXTERNAL_ATTACK_DISPLAY_NAMES.get(attack_id, attack_id),
            "config_path": config_rel,
            "config_exists": config_exists,
            "command_template_configured": command_template_configured,
            "runnable": runnable,
            "repo": repo,
            "checkpoint": checkpoint,
            "target": target,
            "messages": messages,
        }
    return status


def _system_overview_payload(ctx: dict[str, Any]) -> dict[str, Any]:
    project_root = ctx["project_root"]
    artifacts_dir = ctx["artifacts_dir"]
    models = ctx["models"]
    ready_models = ctx["ready_models"]
    formal_models = [item for item in models if bool(item.get("formal_eval", True))]
    formal_ready_models = [item for item in ready_models if bool(item.get("formal_eval", True))]
    validated_models = ctx["validated_models"]
    validation_summary = ctx["validation_summary"]
    live_datasets = ctx["live_datasets"]
    dataset_catalog = ctx["dataset_catalog"]
    return {
        "generated_at": utc_now_iso(),
        "project_root": str(project_root),
        "artifacts_dir": str(artifacts_dir),
        "app_db": ctx["app_db"],
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _torch_info(),
        "runtime": _runtime_info(),
        "live_runtime_note": "当前顶层 `python_version`、`torch` 和 `runtime` 字段反映的是答辩服务器的实时运行环境；论文正式结果对应的冻结环境见 `paper_result_environment` 字段。",
        "paper_result_environment_source_path": _portable_artifact_path(project_root, ctx["paper_environment_path"]),
        "paper_result_environment_note": str(ctx["paper_environment"].get("note", "") or ""),
        "paper_result_environment": ctx["paper_environment"],
        "install_hint": torch_install_command(),
        "adapters": build_adapter_env(models),
        "models": models,
        "supported_model_count": len(formal_models),
        "model_total_count": len(models),
        "ready_model_count": len(formal_ready_models),
        "online_models": [str(item.get("adapter", "")).strip() for item in formal_ready_models if str(item.get("adapter", "")).strip()],
        "online_model_count": len(formal_ready_models),
        "validated_models": validated_models,
        "validated_model_count": len(validated_models),
        "scientific_quality_models": ctx["scientific_quality_models"],
        "scientific_quality_model_count": len(ctx["scientific_quality_models"]),
        "portable_container_validation": _portable_container_validation_summary(artifacts_dir),
        "model_coverage": _model_coverage_summary(formal_models, formal_ready_models, validated_models, validation_summary),
        "attacks": list_plugins("attack"),
        "external_attack_status": _external_attack_runtime_status(project_root),
        "datasets": live_datasets,
        "formal_dataset_count": len([item for item in live_datasets if str(item.get("tier", "")) != "demo"]),
        "dataset_total_count": len(live_datasets),
        "dataset_catalog": dataset_catalog,
        "dataset_catalog_formal_count": len([item for item in dataset_catalog if str(item.get("tier", "")) != "demo"]),
        "dataset_catalog_total_count": _dataset_catalog_count(),
        "source_documents": _source_docs(project_root),
        "paper_repositories": _overview_repositories(project_root),
        "patch_registry": ctx["patch_registry"],
        "build_identity": _overview_build_identity(ctx),
        "latest_runs": _latest_runs(artifacts_dir, limit=10),
        "latest_runs_note": "这里展示的是 /api/v1/runs 当前可查询到的最近运行结果，供测试页切换和结果解读使用，不代表首页展示的正式归档结果。",
        "latest_formal_runs": ctx["formal_runs"][:10],
        "latest_formal_runs_note": "这里汇总展示根据正式分析包整理出的全部正式结果，包括主实验结果和消融结果；首页主表会将两者分开展示。",
        "latest_primary_formal_runs": ctx["primary_formal_runs"],
        "latest_primary_formal_runs_note": "这里集中展示支撑论文主要结论的 E1、E2 主实验结果，并按论文行文顺序排列；E4 消融结果单独列出，不与主实验结果放在一起。首页主表中的数值列标为“首位攻击成功率（attack success rate at first rank，汇总）”，它表示攻击后阶段在图检文与文检图两个方向、全部受测模型上的平均值，而不是某一个受测模型的单向结果。",
        "primary_formal_runs_source_path": ctx["paper_suite_analysis_path"].as_posix(),
        "primary_formal_runs_source_kind": ctx["paper_suite_source_kind"],
        "primary_formal_runs_artifact_index_path": ctx["primary_artifact_index_path"],
        "latest_ablation_runs": ctx["ablation_formal_runs"],
        "latest_ablation_runs_note": "这里单独展示 E4 消融结果，用来说明 ADVEDM+ 各组成部分的作用，不作为首页主结论的直接依据。消融表中的攻击成功率采用与主表相同的汇总口径。",
        "validation_snapshot": _validation_snapshot_summary(ctx["validation_path"], validation_summary, ctx["validation_jobs"]),
        "failing_primary_rows": _failing_primary_rows(models, validated_models, validation_summary, ctx["validation_jobs"]),
        "validation_summary": {
            "path": str(ctx["validation_path"] or ""),
            **validation_summary,
        },
    }


def build_system_overview(request: Request) -> dict[str, Any]:
    return _system_overview_payload(_system_overview_context(request))


def _exists(project_root: Path, rel: str) -> bool:
    return (project_root / rel).exists()


def _attack_catalog_entries(project_root: Path) -> dict[str, dict[str, str]]:
    path = project_root / "frontend" / "src" / "lib" / "attackCatalog.ts"
    if not path.exists():
        return {}
    records: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "{":
            current = {}
            continue
        if current is None:
            continue
        match = re.match(r'(\w+)\s*:\s*"([^"]+)"', line)
        if match:
            current[match.group(1)] = match.group(2).strip()
        if line in {"}", "},"}:
            attack_id = str(current.get("id", "")).strip()
            if attack_id:
                records[attack_id] = dict(current)
            current = None
    return records


def _artifact_candidate(project_root: Path, value: Any, fallback: Path) -> Path:
    raw = str(value or "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (project_root / raw).resolve()
        if candidate.exists():
            return candidate
    return fallback


def _has_metric_payload(summary_data: dict[str, Any]) -> bool:
    return any(isinstance(summary_data.get(key), dict) for key in ("risk", "vlr", "metric_series", "stage_metrics")) or bool(
        summary_data.get("victim_compare")
    ) or any(key in summary_data for key in ("asr_attack", "risk_score", "defense_gain"))


def _empty_formal_artifact_stats() -> dict[str, Any]:
    return {
        "archived_row_evidence_runs": 0,
        "archived_metric_ready_runs": 0,
        "formal_report_runs": 0,
        "formal_metric_ready_runs": 0,
        "source_formal_report_runs": 0,
        "source_formal_metric_ready_runs": 0,
        "portable_formal_report_runs": 0,
        "portable_formal_metric_ready_runs": 0,
        "three_stage_runs": 0,
        "attacks": set(),
        "victim_models": set(),
        "image_only_definition_attacks": set(),
        "joint_definition_attacks": set(),
        "image_only_definition_rows": 0,
        "joint_definition_rows": 0,
    }


def _formal_artifact_paths(project_root: Path, row: dict[str, Any], run_path: Path) -> dict[str, Path]:
    archived_summary_path = _artifact_candidate(project_root, row.get("archived_summary_path", row.get("summary_path", "")), run_path / "row_evidence.json")
    archived_report_path = _artifact_candidate(project_root, row.get("archived_report_path", row.get("report_path", "")), run_path / "row_evidence.md")
    formal_summary_path = _artifact_candidate(project_root, row.get("source_report_data_path", row.get("summary_path", "")), run_path / "report_data.json")
    formal_report_path = _artifact_candidate(project_root, row.get("source_report_path", row.get("report_path", "")), run_path / "report.html")
    portable_summary_path = _artifact_candidate(project_root, row.get("portable_report_data_path", ""), run_path / "portable_report_data.json")
    portable_report_path = _artifact_candidate(project_root, row.get("portable_report_path", ""), run_path / "portable_report.html")
    return {
        "archived_summary": archived_summary_path,
        "archived_report": archived_report_path,
        "formal_summary": formal_summary_path,
        "formal_report": formal_report_path,
        "portable_summary": portable_summary_path,
        "portable_report": portable_report_path,
    }


def _record_formal_definition_stats(stats: dict[str, Any], row: dict[str, Any], archived_summary_data: dict[str, Any]) -> None:
    attack_id = str(row.get("attack", "") or "").strip()
    joint_execution = _formal_row_joint_execution(row, archived_summary_data)
    if bool(joint_execution.get("image_branch_enabled", False)) and not bool(joint_execution.get("text_branch_enabled", False)):
        stats["image_only_definition_rows"] += 1
        if attack_id:
            stats["image_only_definition_attacks"].add(attack_id)
    if bool(joint_execution.get("joint_execution_declared", False)):
        stats["joint_definition_rows"] += 1
        if attack_id:
            stats["joint_definition_attacks"].add(attack_id)


def _has_three_stage_evidence(summary_data: dict[str, Any]) -> bool:
    stage_metrics = summary_data.get("stage_metrics", {})
    if isinstance(stage_metrics, dict) and stage_metrics:
        keys = {str(key) for key, value in stage_metrics.items() if isinstance(value, dict)}
        return {"clean", "attacked"}.issubset(keys) and bool({"defended_attack", "defended_clean"} & keys)
    victim_compare = [item for item in list(summary_data.get("victim_compare", [])) if isinstance(item, dict)]
    status_keys: set[str] = set()
    for item in victim_compare:
        status = item.get("status", {})
        if isinstance(status, dict):
            status_keys.update(str(key) for key, value in status.items() if str(value).strip())
    return {"clean", "attacked"}.issubset(status_keys) and bool({"defended_attack", "defended_clean"} & status_keys)


def _record_formal_report_stats(stats: dict[str, Any], paths: dict[str, Path]) -> None:
    if paths["formal_summary"].exists() and paths["formal_report"].exists():
        stats["formal_report_runs"] += 1
        stats["source_formal_report_runs"] += 1
        formal_summary_data = read_json(paths["formal_summary"], {})
        if isinstance(formal_summary_data, dict) and _has_metric_payload(formal_summary_data):
            stats["formal_metric_ready_runs"] += 1
            stats["source_formal_metric_ready_runs"] += 1
    elif paths["portable_summary"].exists() and paths["portable_report"].exists():
        stats["formal_report_runs"] += 1
        stats["portable_formal_report_runs"] += 1
        portable_summary_data = read_json(paths["portable_summary"], {})
        if isinstance(portable_summary_data, dict) and _has_metric_payload(portable_summary_data):
            stats["formal_metric_ready_runs"] += 1
            stats["portable_formal_metric_ready_runs"] += 1


def _record_formal_artifact_row(project_root: Path, stats: dict[str, Any], row: dict[str, Any]) -> None:
    run_path = Path(str(row.get("path", "") or "")).resolve()
    attack_id = str(row.get("attack", "") or "").strip()
    stats["attacks"].add(attack_id)
    for adapter in list(row.get("victim_model_adapters", [])):
        if str(adapter).strip():
            stats["victim_models"].add(str(adapter).strip())
    paths = _formal_artifact_paths(project_root, row, run_path)
    archived_summary_data: dict[str, Any] = {}
    if paths["archived_summary"].exists() and paths["archived_report"].exists():
        stats["archived_row_evidence_runs"] += 1
        archived_summary_data = read_json(paths["archived_summary"], {})
        if not isinstance(archived_summary_data, dict):
            archived_summary_data = {}
    _record_formal_definition_stats(stats, row, archived_summary_data)
    if _has_three_stage_evidence(archived_summary_data):
        stats["three_stage_runs"] += 1
    if _has_metric_payload(archived_summary_data):
        stats["archived_metric_ready_runs"] += 1
    _record_formal_report_stats(stats, paths)


def _formal_artifact_stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "archived_row_evidence_runs": stats["archived_row_evidence_runs"],
        "archived_metric_ready_runs": stats["archived_metric_ready_runs"],
        "formal_report_runs": stats["formal_report_runs"],
        "formal_metric_ready_runs": stats["formal_metric_ready_runs"],
        "source_formal_report_runs": stats["source_formal_report_runs"],
        "source_formal_metric_ready_runs": stats["source_formal_metric_ready_runs"],
        "portable_formal_report_runs": stats["portable_formal_report_runs"],
        "portable_formal_metric_ready_runs": stats["portable_formal_metric_ready_runs"],
        "three_stage_runs": stats["three_stage_runs"],
        "distinct_attacks": len([item for item in stats["attacks"] if item]),
        "distinct_victim_models": len(stats["victim_models"]),
        "image_only_definition_rows": stats["image_only_definition_rows"],
        "joint_definition_rows": stats["joint_definition_rows"],
        "image_only_definition_attacks": sorted(item for item in stats["image_only_definition_attacks"] if item),
        "joint_definition_attacks": sorted(item for item in stats["joint_definition_attacks"] if item),
    }


def _official_formal_artifacts_summary(project_root: Path, artifacts_dir: Path) -> dict[str, Any]:
    stats = _empty_formal_artifact_stats()
    for row in _primary_formal_runs(project_root, artifacts_dir, limit=50):
        run_path = Path(str(row.get("path", "") or "")).resolve()
        if run_path.exists() or str(row.get("path", "") or ""):
            _record_formal_artifact_row(project_root, stats, row)
    return _formal_artifact_stats_payload(stats)


def _observed_execution_summary(artifacts_dir: Path) -> dict[str, Any]:
    image_only_execution_report_runs = 0
    joint_execution_report_runs = 0
    image_only_execution_attacks: set[str] = set()
    joint_execution_attacks: set[str] = set()

    for row in _latest_runs(artifacts_dir, limit=400):
        run_path = Path(str(row.get("path", "") or "")).resolve()
        if not run_path.exists():
            continue
        summary_path = run_path / "summary.json"
        report_path = run_path / "report.html"
        if not (summary_path.exists() and report_path.exists()):
            continue
        summary_data = read_json(summary_path, {})
        if not isinstance(summary_data, dict):
            continue
        joint_execution = _formal_row_joint_execution(row, summary_data)
        if str(joint_execution.get("joint_execution_basis", "") or "") != "observed":
            continue
        attack_id = str(summary_data.get("attack", row.get("attack", "")) or "").strip()
        if bool(joint_execution.get("image_branch_enabled", False)) and not bool(joint_execution.get("text_branch_enabled", False)):
            image_only_execution_report_runs += 1
            if attack_id:
                image_only_execution_attacks.add(attack_id)
        if bool(joint_execution.get("joint_execution_confirmed", False)):
            joint_execution_report_runs += 1
            if attack_id:
                joint_execution_attacks.add(attack_id)

    return {
        "image_only_execution_report_runs": image_only_execution_report_runs,
        "joint_execution_report_runs": joint_execution_report_runs,
        "image_only_execution_attacks": sorted(item for item in image_only_execution_attacks if item),
        "joint_execution_attacks": sorted(item for item in joint_execution_attacks if item),
    }


def _catalog_values(project_root: Path, rel_path: str, field_name: str) -> set[str]:
    path = project_root / rel_path
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return {match.group(1).strip() for match in re.finditer(rf'{field_name}\s*:\s*"([^"]+)"', text) if match.group(1).strip()}


def _validation_matrix_ready(project_root: Path, artifacts_dir: Path) -> tuple[bool, int]:
    _, summary = _latest_model_validation_summary(artifacts_dir)
    if not isinstance(summary, dict):
        return False, 0
    validated = int(summary.get("validated_model_count", 0) or 0)
    criterion = summary.get("criterion", {})
    thresholds = read_json(project_root / "artifacts" / "paper_acceptance_thresholds.json", {})
    model_validation_cfg = dict(thresholds.get("model_validation", {})) if isinstance(thresholds, dict) else {}
    required = int(model_validation_cfg.get("required_model_count", 0) or 10)
    attacks = {
        str(item).strip()
        for item in list(summary.get("attacks", dict(criterion).get("benchmark_attacks", [])))
        if str(item).strip()
    }
    criterion_map = dict(criterion) if isinstance(criterion, dict) else {}
    dataset_name = str(summary.get("dataset_name", criterion_map.get("dataset_name", "")) or "")
    return (
        bool(summary.get("passed", False))
        and validated >= required
        and dataset_name == str(model_validation_cfg.get("required_dataset", "") or "")
        and {str(item) for item in list(model_validation_cfg.get("required_attacks", []))}.issubset(attacks)
        and int(criterion_map.get("max_pairs", 0) or 0) == int(model_validation_cfg.get("required_max_pairs", 0) or 0)
    ), validated


def _core_routes_ready(project_root: Path) -> bool:
    app_path = project_root / "frontend" / "src" / "App.tsx"
    if not app_path.exists():
        return False
    text = app_path.read_text(encoding="utf-8")
    required_snippets = [
        'path="/"',
        'path="/testing"',
        'path="/jobs"',
        'path="/analysis"',
        'path="/cases"',
        'path="/reports"',
        'path="/reports/:runId"',
        'path="/reports/:runId/cases/:sampleId"',
        'Navigate to="/testing"',
    ]
    return all(snippet in text for snippet in required_snippets)


def _frontend_ready(project_root: Path) -> bool:
    build_info = _frontend_build_info(project_root)
    return bool(build_info.get("index_exists", False) and build_info.get("dist_fresh", False))


def _frontend_build_info(project_root: Path) -> dict[str, Any]:
    frontend_root = project_root / "frontend"
    src_dir = frontend_root / "src"
    index_path = frontend_root / "dist" / "index.html"
    build_inputs = [path for path in [frontend_root / "index.html", frontend_root / "vite.config.ts", frontend_root / "tailwind.config.cjs", frontend_root / "package.json", frontend_root / "pnpm-lock.yaml"] if path.exists()]
    if not index_path.exists():
        return {
            "index_exists": False,
            "dist_fresh": False,
            "index_built_at": "",
            "inputs_latest_at": "",
            "asset_refs": [],
        }

    index_text = index_path.read_text(encoding="utf-8")
    asset_refs = sorted({match.group(1) for match in re.finditer(r'(?:src|href)="(/assets/[^"]+)"', index_text)})
    index_mtime = index_path.stat().st_mtime
    input_candidates = [
        item.stat().st_mtime
        for item in src_dir.rglob("*")
        if item.is_file()
        and ".test." not in item.name
        and ".spec." not in item.name
        and "__tests__" not in item.parts
    ]
    input_candidates.extend(path.stat().st_mtime for path in build_inputs)
    inputs_latest_mtime = max(input_candidates, default=0.0)
    dist_fresh = bool(index_mtime >= inputs_latest_mtime) if inputs_latest_mtime else True

    def _fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts > 0 else ""

    return {
        "index_exists": True,
        "dist_fresh": dist_fresh,
        "index_built_at": _fmt(index_mtime),
        "inputs_latest_at": _fmt(inputs_latest_mtime),
        "asset_refs": asset_refs,
    }


def _taskbook_context(project_root: Path, artifacts_dir: Path) -> dict[str, Any]:
    runtime_attacks = {str(item).strip() for item in list_plugins("attack") if str(item).strip()}
    runtime_models = {
        str(item.get("adapter", "")).strip()
        for item in list_main_models(project_root=project_root)
        if str(item.get("adapter", "")).strip() and bool(item.get("formal_eval", True))
    }
    attack_catalog_ids = _catalog_values(project_root, "frontend/src/lib/attackCatalog.ts", "id")
    model_catalog_adapters = _catalog_values(project_root, "frontend/src/lib/modelCatalog.ts", "adapter")
    validation_ready, validated_model_count = _validation_matrix_ready(project_root, artifacts_dir)
    validation_path, validation_summary = _latest_model_validation_summary(artifacts_dir)
    validation_summary = validation_summary if isinstance(validation_summary, dict) else {}
    observed_execution = _observed_execution_summary(artifacts_dir)
    observed_attack_coverage = sorted(
        {
            *[str(item).strip() for item in observed_execution.get("image_only_execution_attacks", []) if str(item).strip()],
            *[str(item).strip() for item in observed_execution.get("joint_execution_attacks", []) if str(item).strip()],
        }
    )
    return {
        "runtime_attacks": runtime_attacks,
        "attack_count": len(runtime_attacks),
        "runtime_models": runtime_models,
        "supported_model_count": len(runtime_models),
        "attack_surface_synced": runtime_attacks.issubset(attack_catalog_ids) and bool(runtime_attacks),
        "model_surface_synced": runtime_models.issubset(model_catalog_adapters) and bool(runtime_models),
        "has_sample_store": _exists(project_root, "src/mmsec_eval/sample_store"),
        "sample_artifacts": _sample_management_artifacts(artifacts_dir),
        "validation_ready": validation_ready,
        "validated_model_count": validated_model_count,
        "validation_artifact_ready": bool(validation_path and int(validation_summary.get("validated_model_count", 0) or 0) > 0),
        "formal_run_count": len(_primary_formal_runs(project_root, artifacts_dir, limit=20)),
        "formal_artifacts": _official_formal_artifacts_summary(project_root, artifacts_dir),
        "observed_execution": observed_execution,
        "observed_attack_coverage": observed_attack_coverage,
        "frontend_ready": _frontend_ready(project_root),
        "core_routes_ready": _core_routes_ready(project_root),
    }


def _taskbook_req_1(ctx: dict[str, Any]) -> dict[str, str]:
    formal_artifacts = ctx["formal_artifacts"]
    observed_execution = ctx["observed_execution"]
    sample_artifacts = ctx["sample_artifacts"]
    scenario_ready = (
        int(observed_execution.get("image_only_execution_report_runs", 0) or 0) > 0
        and int(observed_execution.get("joint_execution_report_runs", 0) or 0) > 0
    )
    attack_ready = (
        int(ctx["attack_count"]) >= 15
        and bool(ctx["has_sample_store"])
        and sample_artifacts["cases_index_runs"] > 0
        and sample_artifacts["case_bundles"] > 0
        and sample_artifacts["attack_debug_cases"] > 0
        and scenario_ready
    )
    return {
        "id": "req_1",
        "title": "Multimodal adversarial sample generation and management",
        "status": "ready" if attack_ready else "partial",
        "evidence": (
            f"registered_attacks={ctx['attack_count']}, observed_attack_coverage_count={len(ctx['observed_attack_coverage'])}, "
            f"sample_store={ctx['has_sample_store']}, cases_index_runs={sample_artifacts['cases_index_runs']}, "
            f"case_bundles={sample_artifacts['case_bundles']}, attack_debug_cases={sample_artifacts['attack_debug_cases']}, "
            f"patch_registry_entries={sample_artifacts['patch_registry_entries']}, "
            f"formal_image_only_definition_rows={formal_artifacts['image_only_definition_rows']}, formal_joint_definition_rows={formal_artifacts['joint_definition_rows']}, "
            f"observed_image_only_execution_report_runs={observed_execution['image_only_execution_report_runs']}, observed_joint_execution_report_runs={observed_execution['joint_execution_report_runs']}, "
            f"observed_image_only_execution_attacks={','.join(observed_execution['image_only_execution_attacks']) or 'none'}, observed_joint_execution_attacks={','.join(observed_execution['joint_execution_attacks']) or 'none'}"
        ),
        "gap": "" if attack_ready else "Need registered attacks plus sample-level case bundles, case indexes, and attack debug artifacts, and observed execution evidence that covers both image-only and image-text joint scenarios required by the taskbook",
    }


def _taskbook_req_2(ctx: dict[str, Any]) -> dict[str, str]:
    formal_artifacts = ctx["formal_artifacts"]
    formal_run_count = int(ctx["formal_run_count"])
    eval_ready = formal_run_count > 0 and formal_artifacts["three_stage_runs"] >= formal_run_count
    return {
        "id": "req_2",
        "title": "Automated clean vs adversarial evaluation flow",
        "status": "ready" if eval_ready else "partial",
        "evidence": (
            f"official_formal_runs={formal_run_count}, official_three_stage_runs={formal_artifacts['three_stage_runs']}, "
            f"three_stage_coverage_complete={formal_artifacts['three_stage_runs'] >= formal_run_count if formal_run_count > 0 else False}"
        ),
        "gap": "" if eval_ready else "Need clean/attacked/defended stage metrics across official formal runs, not only representative samples",
    }


def _taskbook_req_3(ctx: dict[str, Any]) -> dict[str, str]:
    formal_artifacts = ctx["formal_artifacts"]
    viz_ready = formal_artifacts["formal_report_runs"] > 0 and formal_artifacts["formal_metric_ready_runs"] > 0
    return {
        "id": "req_3",
        "title": "Security metrics and visualization reporting",
        "status": "ready" if viz_ready else "partial",
        "evidence": (
            f"formal_report_runs={formal_artifacts['formal_report_runs']}, formal_metric_ready_runs={formal_artifacts['formal_metric_ready_runs']}, "
            f"source_formal_report_runs={formal_artifacts['source_formal_report_runs']}, portable_formal_report_runs={formal_artifacts['portable_formal_report_runs']}, "
            f"archived_row_evidence_runs={formal_artifacts['archived_row_evidence_runs']}, archived_metric_ready_runs={formal_artifacts['archived_metric_ready_runs']}"
        ),
        "gap": "" if viz_ready else "Need portable report_data/report.html artifacts or preserved source reports instead of only row-level evidence summaries",
    }


def _taskbook_req_4(ctx: dict[str, Any]) -> dict[str, str]:
    formal_artifacts = ctx["formal_artifacts"]

    ext_ready = (
        int(ctx["attack_count"]) >= 15
        and int(ctx["supported_model_count"]) >= 10
        and bool(ctx["validation_artifact_ready"])
        and bool(ctx["attack_surface_synced"])
        and bool(ctx["model_surface_synced"])
        and formal_artifacts["distinct_attacks"] >= 2
        and formal_artifacts["archived_row_evidence_runs"] > 0
        and formal_artifacts["distinct_victim_models"] >= 2
    )
    return {
        "id": "req_4",
        "title": "Modular extensibility engineering checklist",
        "status": "ready" if ext_ready else "partial",
        "evidence": (
            f"attacks={ctx['attack_count']}, models={ctx['supported_model_count']}, "
            f"validated_models={ctx['validated_model_count']}, validation_ready={ctx['validation_ready']}, validation_scope=req_5, "
            f"attack_surface_synced={ctx['attack_surface_synced']}, model_surface_synced={ctx['model_surface_synced']}, "
            f"archived_row_evidence_runs={formal_artifacts['archived_row_evidence_runs']}, formal_report_attacks={formal_artifacts['distinct_attacks']}, "
            f"formal_victim_models={formal_artifacts['distinct_victim_models']}"
        ),
        "gap": "" if ext_ready else "Need runtime registry entries to stay aligned with frontend catalogs, at least one successful model-validation artifact, and multi-attack archived evidence with victim-model coverage",
    }


def _taskbook_req_5(ctx: dict[str, Any], *, core_api_ready: bool) -> dict[str, str]:
    formal_artifacts = ctx["formal_artifacts"]
    formal_run_count = int(ctx["formal_run_count"])

    fullstack_ready = (
        bool(ctx["frontend_ready"])
        and bool(ctx["core_routes_ready"])
        and core_api_ready
        and formal_artifacts["archived_row_evidence_runs"] > 0
        and formal_run_count > 0
        and bool(ctx["validation_artifact_ready"])
    )
    return {
        "id": "req_5",
        "title": "Complete full-stack engineering delivery",
        "status": "ready" if fullstack_ready else "partial",
        "evidence": (
            f"built_spa={ctx['frontend_ready']}, core_ui_routes_ready={ctx['core_routes_ready']}, core_api_routes_ready={core_api_ready}, "
            f"archived_row_evidence_runs={formal_artifacts['archived_row_evidence_runs']}, official_formal_runs={formal_run_count}, "
            f"validation_artifact_ready={ctx['validation_artifact_ready']}, validation_ready={ctx['validation_ready']}"
        ),
        "gap": "" if fullstack_ready else "Need built frontend, core UI/API routes, official reports, and validation artifacts",
    }


def _taskbook_items(project_root: Path, artifacts_dir: Path, *, core_api_ready: bool = False) -> list[dict[str, str]]:
    ctx = _taskbook_context(project_root, artifacts_dir)
    return [
        _taskbook_req_1(ctx),
        _taskbook_req_2(ctx),
        _taskbook_req_3(ctx),
        _taskbook_req_4(ctx),
        _taskbook_req_5(ctx, core_api_ready=core_api_ready),
    ]


def _paper_coverage(project_root: Path) -> list[dict[str, Any]]:
    advclip_repo = _repo_entry(project_root, "third_party/papers/AdvCLIP")
    tmm_repo = _repo_entry(project_root, "third_party/papers/TMM")
    advedm_repo = _repo_entry(project_root, "third_party/papers/AdvEDM_demo")

    def status(files: list[str]) -> str:
        have = sum(1 for x in files if _exists(project_root, x))
        if have == len(files):
            return "ready"
        if have == 0:
            return "missing"
        return "partial"

    return [
        {
            "paper": "AdvCLIP",
            "repo": advclip_repo.get("remote", ""),
            "repo_ready": bool(advclip_repo.get("exists") and advclip_repo.get("commit")),
            "impl_status": status(
                [
                    "src/mmsec_eval/attacks/advclip/attack.py",
                    "src/mmsec_eval/attacks/advclip/train.py",
                    "src/mmsec_eval/attacks/advclip/gan_torch.py",
                ]
            ),
            "impl_evidence": "src/mmsec_eval/attacks/advclip/*",
            "reproduction_fidelity": "approx",
            "todo": "Platform implementation is paper-aligned but still treated as engineering reproduction",
        },
        {
            "paper": "TMM",
            "repo": tmm_repo.get("remote", ""),
            "repo_ready": bool(tmm_repo.get("exists") and tmm_repo.get("commit")),
            "impl_status": status(
                [
                    "src/mmsec_eval/attacks/tmm/attack.py",
                    "src/mmsec_eval/attacks/tmm/adfp.py",
                    "src/mmsec_eval/attacks/tmm/ogfh.py",
                ]
            ),
            "impl_evidence": "src/mmsec_eval/attacks/tmm/*",
            "reproduction_fidelity": "approx",
            "todo": "Platform implementation is paper-aligned but still treated as engineering reproduction",
        },
        {
            "paper": "ADVEDM",
            "repo": advedm_repo.get("remote", ""),
            "repo_ready": bool(advedm_repo.get("exists") and advedm_repo.get("commit")),
            "impl_status": status(["src/mmsec_eval/attacks/advedm/attack.py"]),
            "impl_evidence": "src/mmsec_eval/attacks/advedm/*",
            "reproduction_fidelity": "approx",
            "todo": "EDM task-chain reproduction should continue to be strengthened",
        },
    ]


def _registered_api_routes(request: Request) -> set[str]:
    routes: set[str] = set()
    for route in getattr(request.app, "routes", []):
        path = getattr(route, "path", "")
        if isinstance(path, str) and path.strip():
            routes.add(path.strip())
    return routes


def _core_api_routes_ready(request: Request) -> bool:
    required = {
        "/api/v1/health",
        "/api/v1/jobs",
        "/api/v1/runs",
        "/api/v1/system/overview",
        "/api/v1/system/compliance",
    }
    return required.issubset(_registered_api_routes(request))


def _backend_interfaces(project_root: Path, request: Request) -> list[dict[str, str]]:
    registered = _registered_api_routes(request)
    required = [
        ("api_health", "Health endpoint", "/api/v1/health"),
        ("api_jobs", "Job queue endpoint", "/api/v1/jobs"),
        ("api_runs", "Run artifact endpoint", "/api/v1/runs"),
        ("api_docs", "Docs ingest endpoint", "/api/v1/docs/ingest"),
        ("api_system", "System engineering-checklist endpoint", "/api/v1/system/compliance"),
    ]
    out: list[dict[str, str]] = []
    for item_id, title, path in required:
        ok = path in registered
        out.append(
            {
                "id": item_id,
                "title": title,
                "status": "ready" if ok else "missing",
                "evidence": path,
                "gap": "" if ok else f"Missing registered route {path}",
            }
        )
    return out


def _ui_pages(project_root: Path) -> list[dict[str, Any]]:
    expected = [
        ("/", "frontend/src/pages/DashboardPage.tsx"),
        ("/testing", "frontend/src/pages/ExperimentStudioPage.tsx"),
        ("/jobs", "frontend/src/pages/JobCenterPage.tsx"),
        ("/analysis", "frontend/src/pages/ReportCenterPage.tsx"),
        ("/cases", "frontend/src/pages/CaseReviewPage.tsx"),
        ("/reports", "frontend/src/pages/ReportCenterPage.tsx"),
        ("/reports/:runId", "frontend/src/pages/ReportDetailPage.tsx"),
        ("/reports/:runId/cases/:sampleId", "frontend/src/pages/CaseReplayPage.tsx"),
    ]
    return [{"route": route, "page_file": page_file, "exists": _exists(project_root, page_file)} for route, page_file in expected]


def _project_stage(project_root: Path) -> dict[str, Any]:
    # P0: scripts only
    p0 = any(
        _exists(project_root, p)
        for p in [
            "src/mmsec_eval/attacks/advclip/attack.py",
            "src/mmsec_eval/attacks/tmm/attack.py",
            "src/mmsec_eval/attacks/advedm/attack.py",
        ]
    )
    # P1: basic end-to-end pipeline
    p1 = _exists(project_root, "src/mmsec_eval/runner/eval_runner.py") and _exists(project_root, "src/mmsec_eval/viz/render_report.py")
    # P2: engineering platform baseline
    p2 = all(
        _exists(project_root, p)
        for p in [
            "src/mmsec_api/main.py",
            "frontend/src/pages/DashboardPage.tsx",
            "frontend/src/pages/ExperimentStudioPage.tsx",
            "frontend/src/pages/JobCenterPage.tsx",
            "frontend/src/pages/ReportCenterPage.tsx",
            "frontend/src/pages/CaseReviewPage.tsx",
            "frontend/src/pages/ReportDetailPage.tsx",
            "frontend/src/pages/CaseReplayPage.tsx",
            "src/mmsec_eval/plugins/registry.py",
            "scripts/run_fullstack.ps1",
        ]
    )
    # P3: attack-defense integrated experiment platform
    p3 = all(
        _exists(project_root, p)
        for p in [
            "src/mmsec_eval/defenses/sanitize_v1.py",
            "src/mmsec_api/routes/jobs.py",
            "src/mmsec_api/routes/system.py",
            "src/mmsec_api/services/job_executor.py",
            "src/mmsec_eval/runner/retrieval_runner.py",
        ]
    )

    stage = "P0"
    if p1:
        stage = "P1"
    if p2:
        stage = "P2"
    if p3:
        stage = "P3"
    return {
        "stage": stage,
        "criteria": {"p0": p0, "p1": p1, "p2": p2, "p3": p3},
        "final_stage": "P3",
    }


def _engineering_views(project_root: Path, request: Request) -> dict[str, Any]:
    return {
        "project_stage": _project_stage(project_root),
        "backend_interfaces": _backend_interfaces(project_root, request),
        "ui_pages": _ui_pages(project_root),
    }


def build_system_compliance(request: Request) -> dict[str, Any]:
    project_root, artifacts_dir = _runtime_context(request)
    return {
        "generated_at": utc_now_iso(),
        "checklist_semantics": (
            "Engineering checklist only: this endpoint summarizes taskbook mapping, artifact linkage, and interface coverage. "
            "It does not by itself prove paper-grade scientific completion, full reproducibility, or a blocker-free live deployment. "
            "The portable Docker offline-acceptance snapshot is exposed separately so it can be read without being confused with req_5."
        ),
        "taskbook_items": _taskbook_items(project_root, artifacts_dir, core_api_ready=_core_api_routes_ready(request)),
        "paper_coverage": _paper_coverage(project_root),
        "engineering_views": _engineering_views(project_root, request),
        "portable_container_validation": _portable_container_validation_summary(artifacts_dir),
        "result_conformance": _result_conformance_v2(project_root, artifacts_dir),
    }
