# 文件说明：该文件属于后端接口路由，集中实现 runs 相关逻辑。
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from mmsec_api.deps import get_store
from mmsec_api.schemas.models import CaseDetailResponse, RowsResponse, RunCompareResponse, RunListResponse, RunSummary
from mmsec_api.services.risk_compat import apply_compatible_report_data, apply_compatible_risk, derive_compatible_risk
from mmsec_api.services.run_reader import _created_at_from_run_id, discover_runs_from_artifacts, paginate, read_json, read_jsonl
from mmsec_api.store.sqlite import SQLiteStore

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])


# 中文注释：封装 _artifacts_dir 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _artifacts_dir(request: Request) -> str:
    return str(getattr(request.app.state, "artifacts_dir", "artifacts"))


# 中文注释：封装 _run_dir 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _run_dir(run_id: str, artifacts_dir: str = "artifacts") -> Path:
    return Path(artifacts_dir) / "runs" / run_id


# 中文注释：封装 _to_float 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# 中文注释：封装 _optional_float 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _optional_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# 中文注释：封装 _persisted_case_count_for_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _persisted_case_count_for_run(run_id: str, artifacts_dir: str) -> int:
    if not run_id:
        return 0
    cases_path = _run_dir(run_id, artifacts_dir) / "cases_index.jsonl"
    if not cases_path.exists():
        return 0
    try:
        with cases_path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


# 中文注释：封装 _record 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _record(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


# 中文注释：封装 _is_retired_result_key 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _is_retired_result_key(key: object) -> bool:
    text = str(key).lower()
    return any(token in text for token in ("defense", "defended", "recovery"))


# 中文注释：封装 _without_retired_result_fields 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _without_retired_result_fields(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _without_retired_result_fields(v) for k, v in value.items() if not _is_retired_result_key(k)}
    if isinstance(value, list):
        return [_without_retired_result_fields(item) for item in value]
    return value


# 中文注释：封装 _text_from 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _text_from(value: object) -> str:
    if value is None:
        return ""
    return str(value)


# 中文注释：封装 _extract_attack_text 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _extract_attack_text(output_text: str) -> str:
    marker = "攻击文本："
    if marker not in output_text:
        return ""
    return output_text.split(marker, 1)[1].strip()


# 中文注释：封装 _debug_token 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _debug_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return token[:120] if len(token) > 120 else token


# 中文注释：封装 _resolve_existing_run_path 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _resolve_existing_run_path(run_root: Path, value: object) -> Path | None:
    raw = _text_from(value).strip()
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    if path.is_absolute():
        return path if path.exists() else None

    normalized = raw.replace("\\", "/")
    marker = f"runs/{run_root.name}/"
    if marker in normalized:
        candidate = run_root / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate

    candidate = run_root / normalized
    if candidate.exists():
        return candidate
    return None


# 中文注释：封装 _path_ref 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _path_ref(path: Path) -> str:
    return path.as_posix()


DEBUG_RESULT_MARKERS = ("smoke", "debug", "vram_", "quick", "trial", "toy", "demo", "staged_lifecycle")


# 中文注释：封装 _truthy 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "是", "success", "hit"}


# 中文注释：封装 _result_type_for_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _result_type_for_run(raw: dict[str, object]) -> tuple[str, str]:
    text = " ".join(
        str(raw.get(key) or "")
        for key in ("run_id", "benchmark_tag", "dataset_name", "experiment_id", "suite", "suite_label", "evidence_group")
    ).lower()
    if any(marker in text for marker in DEBUG_RESULT_MARKERS):
        return "debug", "运行标记包含 smoke/debug/vram/quick 等调试特征，仍默认展示，但结论需结合证据置信度。"
    return "formal", "作为正式测评记录展示；页面不会因样本量小而把该 run 从默认视图排除。"


# 中文注释：封装 _confidence_for_count 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _confidence_for_count(count: int, result_type: str) -> tuple[str, str]:
    if count >= 30:
        return "high", f"样本规模 n={count}，适合支撑稳定统计比较。"
    if count >= 5:
        return "medium", f"样本规模 n={count}，可支撑趋势判断，建议继续扩展样本。"
    if count > 0:
        return "low", f"样本规模 n={count}，默认展示为核心记录，但统计置信度低。"
    suffix = "调试记录" if result_type == "debug" else "该 run"
    return "low", f"{suffix}未登记可复盘样本，当前只能作为运行级摘要证据。"


# 中文注释：封装 _summary_for_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _summary_for_run(run_id: str, artifacts_dir: str) -> dict[str, object]:
    summary = read_json(_run_dir(run_id, artifacts_dir) / "summary.json", {})
    return apply_compatible_risk(summary if isinstance(summary, dict) else {})


# 中文注释：封装 _report_data_for_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _report_data_for_run(run_id: str, artifacts_dir: str) -> dict[str, object]:
    report = read_json(_run_dir(run_id, artifacts_dir) / "report_data.json", {})
    return apply_compatible_report_data(report if isinstance(report, dict) else {})


# 中文注释：封装 _source_sample_id 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _source_sample_id(row: dict[str, object], fallback: str) -> str:
    for key in ("source_sample_id", "sample_id", "text_id", "image_id", "case_id", "id"):
        raw = str(row.get(key) or "").strip()
        if raw:
            return raw
    return fallback


# 中文注释：封装 _derived_vlr_case_rows 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _derived_vlr_case_rows(run_id: str, artifacts_dir: str, limit: int = 50) -> list[dict[str, object]]:
    run_root = _run_dir(run_id, artifacts_dir)
    summary = _summary_for_run(run_id, artifacts_dir)
    if str(summary.get("task_kind") or "").strip().lower() != "vlr":
        return []
    report = _report_data_for_run(run_id, artifacts_dir)
    raw_rows: list[dict[str, object]] = []
    vlr = report.get("vlr") if isinstance(report.get("vlr"), dict) else {}
    for source in (report.get("rows_preview"), vlr.get("failure_cases") if isinstance(vlr, dict) else None, read_jsonl(run_root / "results.jsonl")):
        if not isinstance(source, list):
            continue
        for item in source:
            if isinstance(item, dict):
                raw_rows.append(item)
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for idx, row in enumerate(raw_rows):
        src = _source_sample_id(row, str(idx + 1))
        victims = summary.get("victim_model_adapters")
        default_victim = victims[0] if isinstance(victims, list) and victims else summary.get("model_adapter") or "vlr"
        victim = str(row.get("victim") or default_victim)
        query_type = str(row.get("query_type") or row.get("scope") or "vlr")
        sample_id = f"vlr-{_debug_token(victim)}-{_debug_token(query_type)}-{_debug_token(src)}"
        if sample_id in seen:
            continue
        seen.add(sample_id)
        retrieval_hit = _truthy(row.get("judge_success")) if "judge_success" in row else False
        attack_success = not retrieval_hit if "judge_success" in row else False
        top5 = row.get("top5_image_ids") if isinstance(row.get("top5_image_ids"), list) else []
        debug_dirs = _vlr_debug_dirs(run_root, src)
        artifact_status = "partial"
        if any((d / "clean_input.png").exists() for d in debug_dirs) and any(_first_vlr_adv_image(d) for d in debug_dirs):
            artifact_status = "complete"
        out.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "source_sample_id": src,
                "case_kind": "derived_vlr_report_case",
                "task_kind": "vlr",
                "dataset_name": str(summary.get("dataset_name") or ""),
                "benchmark_tag": str(summary.get("benchmark_tag") or ""),
                "model_adapter": str(row.get("victim") or summary.get("model_adapter") or ""),
                "attack": str(summary.get("attack") or ""),
                "risk_level": str(summary.get("risk_level") or ""),
                "risk_score": _to_float(summary.get("risk_score"), 0.0),
                "judge_success": attack_success,
                "retrieval_hit": retrieval_hit,
                "judge_reason": str(row.get("judge_reason") or ""),
                "text": str(row.get("text") or ""),
                "gt_image_id": str(row.get("gt_image_id") or ""),
                "top5_image_ids": top5,
                "artifact_status": artifact_status,
                "created_at": str(summary.get("created_at") or _created_at_from_run_id(run_id)),
            }
        )
        if len(out) >= limit:
            break
    return out


# 中文注释：封装 _case_count_for_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _case_count_for_run(run_id: str, artifacts_dir: str) -> int:
    persisted = _persisted_case_count_for_run(run_id, artifacts_dir)
    if persisted:
        return persisted
    return len(_derived_vlr_case_rows(run_id, artifacts_dir, limit=500))


# 中文注释：封装 _evidence_count_for_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _evidence_count_for_run(row: dict[str, object], artifacts_dir: str) -> int:
    case_count = int(_to_float(row.get("case_count"), 0.0))
    if case_count > 0:
        return case_count
    task_kind = str(row.get("task_kind") or "").lower()
    summary = _summary_for_run(str(row.get("run_id") or ""), artifacts_dir)
    if task_kind in {"vqa", "caption"}:
        for key in ("num_effective", "num_samples"):
            count = int(_to_float(summary.get(key), 0.0))
            if count > 0:
                return count
    pair_count = int(_to_float(row.get("sample_pair_count"), 0.0))
    if pair_count > 0:
        return pair_count
    for key in ("num_effective", "num_samples", "num_images"):
        count = int(_to_float(summary.get(key), 0.0))
        if count > 0:
            return count
    return 0


# 中文注释：封装 _decorate_run_row 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _decorate_run_row(row: dict[str, object], artifacts_dir: str) -> dict[str, object]:
    out = dict(row)
    run_id = str(out.get("run_id") or "")
    out.update(derive_compatible_risk(_summary_for_run(run_id, artifacts_dir), out))
    case_count = _case_count_for_run(run_id, artifacts_dir)
    out["case_count"] = case_count
    result_type, result_type_note = _result_type_for_run(out)
    evidence_count = _evidence_count_for_run(out, artifacts_dir)
    confidence, note = _confidence_for_count(evidence_count, result_type)
    out.update(
        {
            "result_type": result_type,
            "result_type_note": result_type_note,
            "evidence_sample_count": evidence_count,
            "evidence_confidence": confidence,
            "evidence_note": note,
            "has_case_evidence": case_count > 0,
            "artifact_evidence_status": "complete" if case_count > 0 else "summary_only",
        }
    )
    return out


# 中文注释：封装 _vlr_debug_dirs 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _vlr_debug_dirs(run_root: Path, source_sample_id: str) -> list[Path]:
    debug_root = run_root / "attack_debug"
    token = _debug_token(source_sample_id)
    candidates = [debug_root / source_sample_id, debug_root / token, debug_root / f"img_{token}", debug_root / f"txt_{token}"]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(path)
        out.append(path)
    return out


# 中文注释：封装 _first_vlr_adv_image 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _first_vlr_adv_image(debug_dir: Path) -> str:
    for path in sorted(debug_dir.glob("*.png")):
        name = path.name.lower()
        if name == "clean_input.png" or "attention" in name or "mask" in name:
            continue
        return _path_ref(path)
    return ""


# 中文注释：封装 _first_debug_file 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _first_debug_file(debug_dirs: list[Path], patterns: list[str]) -> str:
    for directory in debug_dirs:
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                if path.is_file():
                    return _path_ref(path)
    return ""


# 中文注释：封装 _artifact_state 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _artifact_state(key: str, label: str, value: object, run_root: Path, *, expected: bool = True, disabled: bool = False) -> dict[str, str]:
    resolved = _resolve_existing_run_path(run_root, value)
    if resolved is not None and resolved.is_file():
        return {"key": key, "label": label, "status": "available", "reason": "可查看"}
    if disabled:
        return {"key": key, "label": label, "status": "not_enabled", "reason": "本次未启用"}
    if not expected:
        return {"key": key, "label": label, "status": "not_applicable", "reason": "当前方法不适用"}
    return {"key": key, "label": label, "status": "not_recorded", "reason": "历史运行未记录该证据文件"}


# 中文注释：封装 _artifact_capability 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _artifact_capability(bundle: dict[str, object], run_root: Path, summary: dict[str, object], debug_files: list[str]) -> list[dict[str, str]]:
    refs = _record(bundle.get("artifact_refs"))
    task_kind = str(bundle.get("task_kind") or summary.get("task_kind") or "").lower()
    attack = str(summary.get("attack") or _record(_record(bundle.get("adversarial")).get("metadata")).get("attack") or "").lower()
    outputs = _record(bundle.get("outputs"))
    has_output_diff = bool(_text_from(_record(outputs.get("clean")).get("text")).strip() or _text_from(_record(outputs.get("adv")).get("text")).strip())
    return [
        _artifact_state("clean_image", "原始图像", refs.get("clean_image"), run_root),
        _artifact_state("adv_image", "对抗图像", refs.get("adv_image") or refs.get("attack_visualization"), run_root),
        {"key": "output_diff", "label": "输出差异", "status": "available" if has_output_diff else "missing", "reason": "可查看" if has_output_diff else "未记录输入输出差异"},
        {"key": "debug_files", "label": "调试文件", "status": "available" if debug_files else "missing", "reason": "可查看" if debug_files else "未记录调试文件"},
        _artifact_state("attention_map", "注意力热图", refs.get("attention_map"), run_root, expected=attack in {"advedm", "advedm_plus"}),
        _artifact_state("mask_map", "攻击掩码", refs.get("mask_map"), run_root, expected=attack in {"advedm", "advedm_plus"}),
        _artifact_state("patch_preview", "补丁预览", refs.get("patch_preview"), run_root, expected=attack in {"advclip", "xtransfer_uap"}),
        _artifact_state("cot_trace", "CoT 轨迹", _record(bundle.get("diagnostics")).get("cot_trace"), run_root, expected=task_kind in {"embodied"}),
    ]


# 中文注释：封装 _artifact_status_from_capability 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _artifact_status_from_capability(items: list[dict[str, str]]) -> str:
    blocking = [item for item in items if item.get("status") == "missing" and item.get("key") in {"clean_image", "adv_image", "output_diff"}]
    return "partial" if blocking else "complete"


# 中文注释：封装 _attach_artifact_capability 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _attach_artifact_capability(bundle: dict[str, object], run_root: Path, summary: dict[str, object], debug_files: list[str]) -> dict[str, object]:
    out = dict(bundle)
    items = _artifact_capability(out, run_root, summary, debug_files)
    out["artifact_capability"] = items
    out["artifact_status"] = _artifact_status_from_capability(items)
    return out


# 中文注释：封装 _derived_vlr_case_bundle 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _derived_vlr_case_bundle(run_id: str, sample_id: str, artifacts_dir: str) -> tuple[dict[str, object], dict[str, object]] | None:
    rows = _derived_vlr_case_rows(run_id, artifacts_dir, limit=500)
    row = next((item for item in rows if str(item.get("sample_id")) == sample_id), None)
    if row is None:
        return None
    run_root = _run_dir(run_id, artifacts_dir)
    summary = _summary_for_run(run_id, artifacts_dir)
    source_id = str(row.get("source_sample_id") or sample_id)
    debug_dirs = _vlr_debug_dirs(run_root, source_id)
    clean_image = _first_debug_file(debug_dirs, ["clean_input.png"])
    adv_image = _first_vlr_adv_image(debug_dirs[0]) if debug_dirs else ""
    trace = _first_debug_file(debug_dirs, ["*trace.json", "debug.json", "*_debug.json"])
    debug_files: list[str] = []
    debug_root = run_root / "attack_debug"
    for directory in debug_dirs:
        for item in sorted(directory.glob("*")):
            if item.is_file():
                try:
                    debug_files.append(item.relative_to(debug_root).as_posix())
                except ValueError:
                    debug_files.append(item.name)
    top5 = row.get("top5_image_ids") if isinstance(row.get("top5_image_ids"), list) else []
    attack_success = _truthy(row.get("judge_success"))
    text = str(row.get("text") or "")
    bundle: dict[str, object] = {
        "task_kind": "vlr",
        "sample": {"sample_id": sample_id, "text": text, "target_text": str(row.get("gt_image_id") or ""), "metadata": dict(row)},
        "adversarial": {
            "sample_id": sample_id,
            "text": text,
            "perturbation_l0": 0,
            "perturbation_l2": _to_float(summary.get("avg_l2"), 0.0),
            "perturbation_linf": _to_float(summary.get("avg_linf"), 0.0),
            "metadata": {"source_sample_id": source_id, "attack": str(summary.get("attack") or "")},
        },
        "inputs": {"clean": {"text": text}, "adv": {"text": text}},
        "dataset_tag": str(summary.get("benchmark_tag") or summary.get("dataset_name") or ""),
        "model_tag": str(row.get("model_adapter") or summary.get("model_adapter") or ""),
        "outputs": {
            "clean": {"text": f"目标图像：{row.get('gt_image_id') or '未记录'}", "score": 1.0},
            "adv": {"text": f"攻击后 Top-5：{', '.join(str(x) for x in top5) if top5 else '未记录'}", "score": 0.0 if attack_success else 1.0},
        },
        "metrics": {
            "attack_success": attack_success,
            "retrieval_hit_after_attack": bool(row.get("retrieval_hit")),
            "judge_reason": str(row.get("judge_reason") or ""),
            "top5_image_ids": top5,
            "perturbation_l2": _to_float(summary.get("avg_l2"), 0.0),
            "perturbation_linf": _to_float(summary.get("avg_linf"), 0.0),
        },
        "judge": {"success": attack_success, "reason": "gt_missing_from_top5" if attack_success else "gt_still_in_top5_or_unrecorded"},
        "diagnostics": {"embedding_shift": _to_float(summary.get("avg_l2"), 0.0), "text_diff_score": 0.0},
        "artifact_refs": {"clean_image": clean_image, "adv_image": adv_image, "trace": trace},
        "visual_labels": {"clean": "原始图像", "adv": "对抗图像"},
    }
    bundle = _attach_artifact_capability(bundle, run_root, summary, debug_files)
    return bundle, {"files": debug_files}


# 中文注释：封装 _case_debug_dirs 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _case_debug_dirs(run_root: Path, bundle: dict[str, object], sample_id: str) -> list[Path]:
    debug_root = run_root / "attack_debug"
    if not debug_root.exists():
        return []

    refs = _record(bundle.get("artifact_refs"))
    sample = _record(bundle.get("sample"))
    sample_meta = _record(sample.get("metadata"))
    adversarial = _record(bundle.get("adversarial"))
    adv_meta = _record(adversarial.get("metadata"))
    token = _debug_token(sample_id)

    candidates: list[Path] = [
        debug_root / sample_id,
        debug_root / token,
        debug_root / f"txt_{token}",
        debug_root / f"img_{token}",
    ]

    for key in ("source_image", "image", "image_path", "file_name", "filename"):
        value = _text_from(sample_meta.get(key)).strip()
        if value:
            candidates.append(debug_root / f"img_{_debug_token(value)}")

    for key in (
        "debug_source",
        "attention_debug_path",
        "mask_debug_path",
        "joint_debug_path",
        "patch_preview",
    ):
        resolved = _resolve_existing_run_path(run_root, adv_meta.get(key))
        if resolved is not None:
            candidates.append(resolved.parent)

    for key in ("attention_map", "mask_map", "patch_preview", "attack_debug"):
        resolved = _resolve_existing_run_path(run_root, refs.get(key))
        if resolved is not None:
            candidates.append(resolved.parent)

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(resolved)
        out.append(path)
    return out


# 中文注释：封装 _first_existing_ref 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _first_existing_ref(run_root: Path, values: list[object]) -> str:
    for value in values:
        resolved = _resolve_existing_run_path(run_root, value)
        if resolved is not None and resolved.is_file():
            return _path_ref(resolved)
    return ""


# 中文注释：封装 _find_debug_artifact 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _find_debug_artifact(debug_dirs: list[Path], filenames: list[str]) -> str:
    for directory in debug_dirs:
        for filename in filenames:
            candidate = directory / filename
            if candidate.exists() and candidate.is_file():
                return _path_ref(candidate)
    return ""


# 中文注释：封装 _enrich_case_bundle_visual_refs 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _enrich_case_bundle_visual_refs(bundle: dict[str, object], run_root: Path, sample_id: str) -> dict[str, object]:
    refs = dict(_record(bundle.get("artifact_refs")))
    adversarial = _record(bundle.get("adversarial"))
    adv_meta = _record(adversarial.get("metadata"))
    debug_dirs = _case_debug_dirs(run_root, bundle, sample_id)

    if not _text_from(refs.get("attention_map")).strip():
        refs["attention_map"] = _first_existing_ref(
            run_root,
            [adv_meta.get("attention_debug_path"), adv_meta.get("attention_map")],
        ) or _find_debug_artifact(debug_dirs, ["advedm_plus_attention.png", "advedm_attention.png", "attention_map.png", "attention.png"])

    if not _text_from(refs.get("mask_map")).strip():
        refs["mask_map"] = _first_existing_ref(
            run_root,
            [adv_meta.get("mask_debug_path"), adv_meta.get("mask_map")],
        ) or _find_debug_artifact(debug_dirs, ["advedm_plus_mask.png", "advedm_mask.png", "mask_map.png", "mask.png"])

    if not _text_from(refs.get("patch_preview")).strip():
        refs["patch_preview"] = _first_existing_ref(
            run_root,
            [adv_meta.get("patch_preview")],
        ) or _find_debug_artifact(debug_dirs, ["advclip_patch_preview.png", "patch_preview.png"])

    bundle["artifact_refs"] = refs
    return bundle


# 中文注释：封装 _enrich_case_bundle_inputs 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _enrich_case_bundle_inputs(bundle: dict[str, object], run_root: Path, sample_id: str) -> dict[str, object]:
    inputs = dict(_record(bundle.get("inputs")))
    outputs = _record(bundle.get("outputs"))
    sample = _record(bundle.get("sample"))
    adversarial = _record(bundle.get("adversarial"))

    clean_input = dict(_record(inputs.get("clean")))
    adv_input = dict(_record(inputs.get("adv")))

    clean_text = _text_from(clean_input.get("text") or sample.get("text")).strip()
    adv_output = _record(outputs.get("adv"))
    adv_text = _text_from(adv_input.get("text") or adversarial.get("text") or _extract_attack_text(_text_from(adv_output.get("text")))).strip()

    inputs = {
        "clean": {**clean_input, "text": clean_text},
        "adv": {**adv_input, "text": adv_text},
    }
    bundle["inputs"] = inputs
    return bundle


# 中文注释：封装 _normalize_run_row 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _normalize_run_row(raw: dict[str, object]) -> dict[str, object]:
    asr = _to_float(raw.get("asr"), 0.0)
    return {
        "run_id": str(raw.get("run_id") or ""),
        "created_at": str(raw.get("created_at") or ""),
        "task_kind": str(raw.get("task_kind") or ""),
        "dataset_name": str(raw.get("dataset_name") or ""),
        "benchmark_tag": str(raw.get("benchmark_tag") or ""),
        "attack": str(raw.get("attack") or ""),
        "attack_modality": str(raw.get("attack_modality") or ""),
        "eval_scope": str(raw.get("eval_scope") or ""),
        "mode": str(raw.get("mode") or ""),
        "experiment_id": str(raw.get("experiment_id") or ""),
        "suite": str(raw.get("suite") or ""),
        "suite_label": str(raw.get("suite_label") or ""),
        "evidence_group": str(raw.get("evidence_group") or ""),
        "experiment_label": str(raw.get("experiment_label") or ""),
        "model_adapter": str(raw.get("model_adapter") or ""),
        "surrogate_model_adapter": str(raw.get("surrogate_model_adapter") or ""),
        "victim_model_adapters": raw.get("victim_model_adapters") if isinstance(raw.get("victim_model_adapters"), list) else [],
        "asr": asr,
        "asr_attack": _to_float(raw.get("asr_attack"), asr),
        "metric_label": str(raw.get("metric_label") or ""),
        "k_value": int(_to_float(raw.get("k_value"), 0.0)),
        "retrieval_direction_scope": str(raw.get("retrieval_direction_scope") or ""),
        "victim_aggregation": str(raw.get("victim_aggregation") or ""),
        "sample_pair_count": int(_to_float(raw.get("sample_pair_count"), 0.0)),
        "metric_note": str(raw.get("metric_note") or ""),
        "risk_score": _to_float(raw.get("risk_score"), 0.0),
        "risk_level": str(raw.get("risk_level") or ""),
        "risk_scenario": str(raw.get("risk_scenario") or ""),
        "avg_l2": _to_float(raw.get("avg_l2"), 0.0),
        "avg_linf": _to_float(raw.get("avg_linf"), 0.0),
        "clean_r1_mean": _optional_float(raw.get("clean_r1_mean")),
        "attacked_r1_mean": _optional_float(raw.get("attacked_r1_mean")),
        "attack_drop_r1_mean": _optional_float(raw.get("attack_drop_r1_mean")),
        "clean_mean_rank": _optional_float(raw.get("clean_mean_rank")),
        "attacked_mean_rank": _optional_float(raw.get("attacked_mean_rank")),
        "rank_delta_mean": _optional_float(raw.get("rank_delta_mean")),
        "clean_accuracy": _optional_float(raw.get("clean_accuracy")),
        "attacked_accuracy": _optional_float(raw.get("attacked_accuracy")),
        "answer_change_rate": _optional_float(raw.get("answer_change_rate")),
        "target_flip_rate": _optional_float(raw.get("target_flip_rate")),
        "semantic_preservation_rate": _optional_float(raw.get("semantic_preservation_rate")),
        "caption_text_similarity": _optional_float(raw.get("caption_text_similarity")),
        "object_jaccard": _optional_float(raw.get("object_jaccard")),
        "semantic_preservation_score": _to_float(raw.get("semantic_preservation_score"), 0.0),
        "case_count": int(_to_float(raw.get("case_count"), 0.0)),
        "evidence_sample_count": int(_to_float(raw.get("evidence_sample_count"), 0.0)),
        "evidence_confidence": str(raw.get("evidence_confidence") or ""),
        "evidence_note": str(raw.get("evidence_note") or ""),
        "result_type": str(raw.get("result_type") or "formal"),
        "result_type_note": str(raw.get("result_type_note") or ""),
        "has_case_evidence": bool(raw.get("has_case_evidence", False)),
        "artifact_evidence_status": str(raw.get("artifact_evidence_status") or "unknown"),
        "path": str(raw.get("path") or ""),
    }


# 中文注释：封装 _run_evidence_exists 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _run_evidence_exists(raw: dict[str, object], artifacts_dir: str) -> bool:
    run_id = str(raw.get("run_id") or "").strip()
    if not run_id:
        return False
    candidates: list[Path] = []
    cache_path = str(raw.get("path") or "").strip()
    if cache_path:
        candidates.append(Path(cache_path))
    candidates.append(_run_dir(run_id, artifacts_dir))
    for root in candidates:
        if not root.is_absolute():
            root = Path(root)
        if root.exists() and root.is_dir() and ((root / "summary.json").exists() or (root / "report_data.json").exists()):
            return True
    return False


GENERATED_ONLY_RESULT_TYPES = {"generated_only", "sample_generation_only", "pending_evaluation"}
FAKE_MODEL_ADAPTERS = {"fixture_vlm", "dummy"}


# 中文注释：封装 _run_model_values 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _run_model_values(raw: dict[str, object]) -> list[str]:
    values: list[str] = []
    for key in ("model_adapter", "surrogate_model_adapter", "model_tag"):
        text = str(raw.get(key) or "").strip()
        if text:
            values.append(text)
    victims = raw.get("victim_model_adapters")
    if isinstance(victims, list):
        values.extend(str(item).strip() for item in victims if str(item or "").strip())
    elif str(victims or "").strip():
        values.append(str(victims).strip())
    return values


# 中文注释：封装 _has_fake_model_marker 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _has_fake_model_marker(raw: dict[str, object]) -> bool:
    model_values = {item.lower() for item in _run_model_values(raw)}
    if model_values.intersection(FAKE_MODEL_ADAPTERS):
        return True
    model_text = " ".join(model_values)
    return "内置生成式演示模型" in model_text or "演示适配器" in model_text


# 中文注释：封装 _run_root_candidates 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _run_root_candidates(raw: dict[str, object], artifacts_dir: str) -> list[Path]:
    run_id = str(raw.get("run_id") or "").strip()
    candidates: list[Path] = []
    cache_path = str(raw.get("path") or "").strip()
    if cache_path:
        candidates.append(Path(cache_path))
    if run_id:
        candidates.append(_run_dir(run_id, artifacts_dir))
    out: list[Path] = []
    seen: set[str] = set()
    for root in candidates:
        key = root.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


# 中文注释：封装 _has_generated_only_marker 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _has_generated_only_marker(raw: dict[str, object]) -> bool:
    if _truthy(raw.get("sample_generation_only")) or _truthy(raw.get("generation_only")):
        return True
    if _truthy(raw.get("requires_evaluation")):
        return True
    result_type = str(raw.get("result_type") or "").strip().lower()
    if result_type in GENERATED_ONLY_RESULT_TYPES:
        return True
    status = str(raw.get("reusable_status") or raw.get("artifact_status") or "").strip().lower()
    return status == "generated_only"


# 中文注释：封装 _is_generated_only_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _is_generated_only_run(raw: dict[str, object], artifacts_dir: str) -> bool:
    if _has_generated_only_marker(raw):
        return True
    for root in _run_root_candidates(raw, artifacts_dir):
        if not root.exists() or not root.is_dir():
            continue
        summary = read_json(root / "summary.json", {})
        if isinstance(summary, dict) and _has_generated_only_marker(summary):
            return True
        report = read_json(root / "report_data.json", {})
        if isinstance(report, dict):
            if _has_generated_only_marker(report):
                return True
            report_summary = report.get("summary")
            if isinstance(report_summary, dict) and _has_generated_only_marker(report_summary):
                return True
    return False


# 中文注释：封装 _is_fake_model_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _is_fake_model_run(raw: dict[str, object], artifacts_dir: str) -> bool:
    if _has_fake_model_marker(raw):
        return True
    for root in _run_root_candidates(raw, artifacts_dir):
        if not root.exists() or not root.is_dir():
            continue
        summary = read_json(root / "summary.json", {})
        if isinstance(summary, dict) and _has_fake_model_marker(summary):
            return True
        report = read_json(root / "report_data.json", {})
        if isinstance(report, dict):
            if _has_fake_model_marker(report):
                return True
            report_summary = report.get("summary")
            if isinstance(report_summary, dict) and _has_fake_model_marker(report_summary):
                return True
    return False


# 中文注释：封装 _raise_if_generated_only_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _raise_if_generated_only_run(run_id: str, artifacts_dir: str) -> None:
    if _is_generated_only_run({"run_id": run_id}, artifacts_dir):
        raise HTTPException(status_code=404, detail="pending sample batch has no evaluation result")


# 中文注释：封装 _raise_if_unservable_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _raise_if_unservable_run(run_id: str, artifacts_dir: str) -> None:
    run_root = _run_dir(run_id, artifacts_dir)
    if not run_root.exists() or not run_root.is_dir() or not ((run_root / "summary.json").exists() or (run_root / "report_data.json").exists()):
        raise HTTPException(status_code=404, detail="run not found")
    _raise_if_generated_only_run(run_id, artifacts_dir)
    if _is_fake_model_run({"run_id": run_id}, artifacts_dir):
        raise HTTPException(status_code=404, detail="demo model result has been removed")


# 中文注释：封装 _run_created_sort_value 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _run_created_sort_value(item: dict[str, object]) -> float:
    raw = str(item.get("created_at") or "").strip()
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            raw = ""
    run_id = str(item.get("run_id") or "").strip()
    try:
        return datetime.strptime("".join(run_id.split("_")[:2]), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return 0.0


# 中文注释：封装 _merge_run_rows 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _merge_run_rows(cache_rows: list[dict[str, object]], artifact_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for row in artifact_rows:
        run_id = str(row.get("run_id") or "").strip()
        if run_id:
            merged[run_id] = dict(row)
    for row in cache_rows:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        current = dict(merged.get(run_id, {}))
        for key, value in row.items():
            if value not in ("", None):
                current[key] = value
        merged[run_id] = current
    return sorted(
        merged.values(),
        key=_run_created_sort_value,
        reverse=True,
    )


# 中文注释：封装 _run_model_key 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _run_model_key(row: dict[str, object]) -> str:
    victims = row.get("victim_model_adapters")
    if isinstance(victims, list) and victims:
        return str(victims[0] or "").strip()
    return str(row.get("model_adapter") or "").strip()


# 中文注释：封装 _risk_bucket 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _risk_bucket(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"critical", "high"} or "高" in text:
        return "high"
    if text in {"medium", "moderate", "mid"} or "中" in text:
        return "medium"
    return "low"


# 中文注释：封装 _model_risk_groups 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _model_risk_groups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        model_key = _run_model_key(row)
        if model_key:
            grouped.setdefault(model_key, []).append(row)

    out: list[dict[str, object]] = []
    for model_key, scoped in sorted(grouped.items()):
        scores = [_to_float(row.get("risk_score"), 0.0) for row in scoped]
        buckets = Counter(_risk_bucket(row.get("risk_level")) for row in scoped)
        out.append(
            {
                "model_adapter": model_key,
                "count": len(scoped),
                "avg_risk_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
                "max_risk_score": round(max(scores), 6) if scores else 0.0,
                "high_risk_count": buckets.get("high", 0),
                "medium_risk_count": buckets.get("medium", 0),
                "low_risk_count": buckets.get("low", 0),
                "low_confidence_count": sum(1 for row in scoped if str(row.get("evidence_confidence")) == "low"),
                "debug_count": sum(1 for row in scoped if str(row.get("result_type") or "formal") == "debug"),
            }
        )
    return sorted(out, key=lambda item: (-float(item["avg_risk_score"]), str(item["model_adapter"])))


# 中文注释：封装 _is_evaluation_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _is_evaluation_run(raw: dict[str, object]) -> bool:
    run_id = str(raw.get("run_id") or "").strip().lower()
    benchmark_tag = str(raw.get("benchmark_tag") or "").strip().lower()
    model_adapter = str(raw.get("model_adapter") or "").strip().lower()
    if _has_fake_model_marker(raw):
        return False
    if run_id.endswith("_demo") or benchmark_tag == "seed_bootstrap" or model_adapter == "dummy":
        return False
    task_kind = str(raw.get("task_kind") or "").strip().lower()
    return task_kind not in {"advclip_train", "train_advclip"}


# 中文注释：封装 _is_demo_like_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _is_demo_like_run(row: dict[str, object]) -> bool:
    run_id = str(row.get("run_id") or "").lower()
    dataset = f"{row.get('dataset_name') or ''} {row.get('benchmark_tag') or ''}".lower()
    victims = row.get("victim_model_adapters") if isinstance(row.get("victim_model_adapters"), list) else []
    model_text = f"{row.get('model_adapter') or ''} {row.get('surrogate_model_adapter') or ''} {' '.join(str(x) for x in victims)}".lower()
    return (
        "demo" in run_id
        or "fixture_vlm" in model_text
        or "fixture" in model_text
        or "内置生成式演示模型" in model_text
        or "dummy" in model_text.split()
        or "seed_bootstrap" in dataset
        or "toy" in dataset
        or "demo" in dataset
        or "演示" in dataset
    )


# 中文注释：封装 _match_text 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _match_text(value: object, needle: str) -> bool:
    return needle.lower() in str(value or "").lower()


# 中文注释：封装 _run_model_search_text 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _run_model_search_text(row: dict[str, object]) -> str:
    victims = row.get("victim_model_adapters") if isinstance(row.get("victim_model_adapters"), list) else []
    return " ".join(str(x or "") for x in [row.get("model_adapter"), row.get("surrogate_model_adapter"), *victims])


# 中文注释：封装 _matches_run_filters 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _matches_run_filters(
    row: dict[str, object],
    *,
    task_kind: str = "",
    dataset: str = "",
    model: str = "",
    attack: str = "",
    result_type: str = "",
    confidence: str = "",
    search: str = "",
    exclude_demo: bool = False,
) -> bool:
    if exclude_demo and _is_demo_like_run(row):
        return False
    if task_kind and str(row.get("task_kind") or "") != task_kind:
        return False
    if attack and str(row.get("attack") or "") != attack:
        return False
    if result_type and str(row.get("result_type") or "") != result_type:
        return False
    if confidence and str(row.get("evidence_confidence") or "") != confidence:
        return False
    if dataset and not (_match_text(row.get("dataset_name"), dataset) or _match_text(row.get("benchmark_tag"), dataset)):
        return False
    if model and not _match_text(_run_model_search_text(row), model):
        return False
    if search:
        hay = " ".join(
            str(row.get(key) or "")
            for key in ("run_id", "dataset_name", "benchmark_tag", "attack", "task_kind", "risk_level", "result_type")
        )
        hay = f"{hay} {_run_model_search_text(row)}"
        if search.lower() not in hay.lower():
            return False
    return True


# 中文注释：封装 _run_sort_value 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _run_sort_value(row: dict[str, object], key: str) -> tuple[int, object, str]:
    if key == "created":
        return (0, str(row.get("created_at") or row.get("run_id") or ""), str(row.get("run_id") or ""))
    if key == "sample_confidence":
        return (0, _to_float(row.get("evidence_sample_count") or row.get("case_count"), 0.0), str(row.get("evidence_confidence") or ""))
    if key == "metric":
        task = str(row.get("task_kind") or "")
        if task == "vqa":
            value = row.get("clean_accuracy")
        elif task == "caption":
            value = row.get("object_jaccard") or row.get("caption_text_similarity") or row.get("semantic_preservation_rate")
        else:
            value = row.get("clean_r1_mean")
        return (0, _to_float(value, -1.0), str(row.get("run_id") or ""))
    if key == "asr_risk":
        return (0, _to_float(row.get("risk_score"), 0.0), _to_float(row.get("asr_attack", row.get("asr")), 0.0))
    text_keys = {
        "run_id": "run_id",
        "result_type": "result_type",
        "model": "model_adapter",
        "attack": "attack",
        "detail": "run_id",
    }
    if key == "task_dataset":
        return (0, f"{row.get('task_kind') or ''} {row.get('dataset_name') or ''} {row.get('benchmark_tag') or ''}", str(row.get("run_id") or ""))
    if key in text_keys:
        return (0, str(row.get(text_keys[key]) or ""), str(row.get("run_id") or ""))
    return (1, str(row.get("created_at") or row.get("run_id") or ""), str(row.get("run_id") or ""))


# 中文注释：封装 _sort_run_rows 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _sort_run_rows(rows: list[dict[str, object]], sort_by: str, sort_dir: str) -> list[dict[str, object]]:
    return [
        row
        for _, row in sorted(
            enumerate(rows),
            key=lambda item: (_run_sort_value(item[1], sort_by), item[0]),
            reverse=(sort_dir != "asc"),
        )
    ]


# 中文注释：封装 _filtered_decorated_rows 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _filtered_decorated_rows(
    artifacts_dir: str,
    store: SQLiteStore,
    *,
    task_kind: str = "",
    dataset: str = "",
    model: str = "",
    attack: str = "",
    result_type: str = "",
    confidence: str = "",
    search: str = "",
    exclude_demo: bool = False,
) -> list[dict[str, object]]:
    return [
        row
        for row in _decorated_rows_all(artifacts_dir, store)
        if _matches_run_filters(
            row,
            task_kind=task_kind,
            dataset=dataset,
            model=model,
            attack=attack,
            result_type=result_type,
            confidence=confidence,
            search=search,
            exclude_demo=exclude_demo,
        )
    ]


# 中文注释：处理 list_runs 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("", response_model=RunListResponse)
def list_runs(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    sort_by: str = Query(default="created"),
    sort_dir: str = Query(default="desc"),
    task_kind: str = Query(default=""),
    dataset: str = Query(default=""),
    model: str = Query(default=""),
    attack: str = Query(default=""),
    result_type: str = Query(default=""),
    confidence: str = Query(default=""),
    search: str = Query(default=""),
    exclude_demo: bool = Query(default=False),
    store: SQLiteStore = Depends(get_store),
) -> RunListResponse:
    artifacts_dir = _artifacts_dir(request)
    rows_all = _filtered_decorated_rows(
        artifacts_dir,
        store,
        task_kind=task_kind,
        dataset=dataset,
        model=model,
        attack=attack,
        result_type=result_type,
        confidence=confidence,
        search=search,
        exclude_demo=exclude_demo,
    )
    total2, rows2 = paginate(_sort_run_rows(rows_all, sort_by, sort_dir), page=page, page_size=page_size)
    items = [RunSummary(**row) for row in rows2]
    return RunListResponse(total=total2, page=page, page_size=page_size, items=items)

# 中文注释：封装 _decorated_rows_all 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _decorated_rows_all(artifacts_dir: str, store: SQLiteStore) -> list[dict[str, object]]:
    total, rows = store.list_runs_cache(page=1, page_size=10000)
    del total
    cache_rows = [
        row
        for row in rows
        if _run_evidence_exists(row, artifacts_dir)
        and _is_evaluation_run(row)
        and not _is_generated_only_run(row, artifacts_dir)
        and not _is_fake_model_run(row, artifacts_dir)
    ]
    artifact_rows = [
        row
        for row in discover_runs_from_artifacts(artifacts_dir)
        if _is_evaluation_run(row) and not _is_generated_only_run(row, artifacts_dir) and not _is_fake_model_run(row, artifacts_dir)
    ]
    merged = _merge_run_rows(cache_rows, artifact_rows)
    return [_decorate_run_row(_normalize_run_row(row if isinstance(row, dict) else {}), artifacts_dir) for row in merged]


# 中文注释：封装 _case_rows_for_run 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _case_rows_for_run(run: dict[str, object], artifacts_dir: str) -> list[dict[str, object]]:
    run_id = str(run.get("run_id") or "")
    run_root = _run_dir(run_id, artifacts_dir)
    rows = read_jsonl(run_root / "cases_index.jsonl")
    if not rows:
        rows = _derived_vlr_case_rows(run_id, artifacts_dir, limit=500)
    out: list[dict[str, object]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item.setdefault("run_id", run_id)
        item.setdefault("task_kind", run.get("task_kind", ""))
        item.setdefault("dataset_name", run.get("dataset_name", ""))
        item.setdefault("benchmark_tag", run.get("benchmark_tag", ""))
        item.setdefault("model_adapter", (run.get("victim_model_adapters") or [run.get("model_adapter", "")])[0] if isinstance(run.get("victim_model_adapters"), list) and run.get("victim_model_adapters") else run.get("model_adapter", ""))
        item.setdefault("attack", run.get("attack", ""))
        if item.get("risk_level") and item.get("risk_level") != run.get("risk_level"):
            item.setdefault("sample_risk_level", item.get("risk_level"))
        if item.get("risk_score") and item.get("risk_score") != run.get("risk_score"):
            item.setdefault("sample_risk_score", item.get("risk_score"))
        item["risk_level"] = run.get("risk_level", "")
        item["risk_score"] = run.get("risk_score", 0.0)
        item.setdefault("created_at", run.get("created_at", ""))
        item.setdefault("result_type", run.get("result_type", "formal"))
        item.setdefault("evidence_confidence", run.get("evidence_confidence", ""))
        item.setdefault("evidence_sample_count", run.get("evidence_sample_count", 0))
        sample_id = str(item.get("sample_id") or "")
        if sample_id and "artifact_status" not in item:
            bundle = read_json(run_root / "cases" / sample_id / "case_bundle.json", {})
            debug_files: list[str] = []
            if isinstance(bundle, dict) and bundle:
                summary = _summary_for_run(run_id, artifacts_dir)
                capability = _artifact_capability(bundle, run_root, summary, debug_files)
                item["artifact_status"] = _artifact_status_from_capability(capability)
            else:
                item["artifact_status"] = "partial"
        out.append(item)
    return out


# 中文注释：处理 run_analytics 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/analytics")
def run_analytics(
    request: Request,
    task_kind: str = Query(default=""),
    dataset: str = Query(default=""),
    model: str = Query(default=""),
    attack: str = Query(default=""),
    result_type: str = Query(default=""),
    confidence: str = Query(default=""),
    search: str = Query(default=""),
    exclude_demo: bool = Query(default=False),
    store: SQLiteStore = Depends(get_store),
):
    artifacts_dir = _artifacts_dir(request)
    rows = _filtered_decorated_rows(
        artifacts_dir,
        store,
        task_kind=task_kind,
        dataset=dataset,
        model=model,
        attack=attack,
        result_type=result_type,
        confidence=confidence,
        search=search,
        exclude_demo=exclude_demo,
    )
    task_counts = Counter(str(row.get("task_kind") or "unknown") for row in rows)
    risk_counts = Counter(str(row.get("risk_level") or "unknown") for row in rows)
    result_type_counts = Counter(str(row.get("result_type") or "formal") for row in rows)
    confidence_counts = Counter(str(row.get("evidence_confidence") or "unknown") for row in rows)
    case_total = sum(int(_to_float(row.get("case_count"), 0.0)) for row in rows)

    # 中文注释：处理 avg 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    task_groups: list[dict[str, object]] = []
    for task, count in sorted(task_counts.items()):
        scoped = [row for row in rows if str(row.get("task_kind") or "unknown") == task]
        task_groups.append(
            {
                "task_kind": task,
                "count": count,
                "case_count": sum(int(_to_float(row.get("case_count"), 0.0)) for row in scoped),
                "avg_asr": avg([_to_float(row.get("asr_attack", row.get("asr")), 0.0) for row in scoped]),
                "avg_risk_score": avg([_to_float(row.get("risk_score"), 0.0) for row in scoped]),
                "low_confidence_count": sum(1 for row in scoped if str(row.get("evidence_confidence")) == "low"),
            }
        )

    attack_matrix: list[dict[str, object]] = []
    for (task, attack_id), count in sorted(Counter((str(row.get("task_kind") or "unknown"), str(row.get("attack") or "unknown")) for row in rows).items()):
        attack_matrix.append({"task_kind": task, "attack": attack_id, "count": count})

    return {
        "total_runs": len(rows),
        "total_cases": case_total,
        "avg_asr_attack": avg([_to_float(row.get("asr_attack", row.get("asr")), 0.0) for row in rows]),
        "formal_runs": result_type_counts.get("formal", 0),
        "debug_runs": result_type_counts.get("debug", 0),
        "high_risk_runs": sum(value for key, value in risk_counts.items() if str(key).lower() in {"critical", "high"} or "高" in str(key)),
        "runs_with_case_evidence": sum(1 for row in rows if _truthy(row.get("has_case_evidence"))),
        "low_confidence_runs": confidence_counts.get("low", 0),
        "task_groups": task_groups,
        "model_risk_groups": _model_risk_groups(rows),
        "risk_distribution": [{"key": key, "count": value} for key, value in sorted(risk_counts.items())],
        "result_type_distribution": [{"key": key, "count": value} for key, value in sorted(result_type_counts.items())],
        "confidence_distribution": [{"key": key, "count": value} for key, value in sorted(confidence_counts.items())],
        "attack_matrix": attack_matrix,
        "latest_runs": rows[:50],
    }


# 中文注释：封装 _option_counter 的内部步骤，让后端接口路由主流程保持清晰并隔离边界细节。
def _option_counter(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    counts = Counter(str(row.get(key) or "") for row in rows if str(row.get(key) or "").strip())
    return [{"key": key, "value": value, "count": count} for value, count in sorted(counts.items())]


# 中文注释：处理 run_options 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/options")
def run_options(
    request: Request,
    exclude_demo: bool = Query(default=False),
    store: SQLiteStore = Depends(get_store),
):
    rows = _filtered_decorated_rows(_artifacts_dir(request), store, exclude_demo=exclude_demo)
    return {
        "task_kinds": _option_counter(rows, "task_kind"),
        "attacks": _option_counter(rows, "attack"),
        "risk_levels": _option_counter(rows, "risk_level"),
        "result_types": _option_counter(rows, "result_type"),
        "confidences": _option_counter(rows, "evidence_confidence"),
    }

# 中文注释：处理 list_case_index 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/cases", response_model=RowsResponse)
def list_case_index(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=500),
    sort_by: str = Query(default="created"),
    sort_dir: str = Query(default="desc"),
    task_kind: str = Query(default=""),
    dataset: str = Query(default=""),
    model: str = Query(default=""),
    attack: str = Query(default=""),
    success: str = Query(default=""),
    risk_level: str = Query(default=""),
    result_type: str = Query(default=""),
    confidence: str = Query(default=""),
    artifact_status: str = Query(default=""),
    search: str = Query(default=""),
    exclude_demo: bool = Query(default=False),
    store: SQLiteStore = Depends(get_store),
) -> RowsResponse:
    artifacts_dir = _artifacts_dir(request)
    runs = _filtered_decorated_rows(artifacts_dir, store, exclude_demo=exclude_demo)
    rows: list[dict[str, object]] = []
    for run in runs:
        rows.extend(_case_rows_for_run(run, artifacts_dir))

    # 中文注释：处理 match_text 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
    def match_text(value: object, needle: str) -> bool:
        return needle.lower() in str(value or "").lower()

    filtered: list[dict[str, object]] = []
    for row in rows:
        if task_kind and str(row.get("task_kind") or "") != task_kind:
            continue
        if dataset and not (match_text(row.get("dataset_name"), dataset) or match_text(row.get("benchmark_tag"), dataset)):
            continue
        if model and not match_text(row.get("model_adapter"), model):
            continue
        if attack and str(row.get("attack") or "") != attack:
            continue
        if risk_level and str(row.get("risk_level") or "") != risk_level:
            continue
        if result_type and str(row.get("result_type") or "") != result_type:
            continue
        if confidence and str(row.get("evidence_confidence") or "") != confidence:
            continue
        if artifact_status and str(row.get("artifact_status") or "") != artifact_status:
            continue
        if success == "success" and not _truthy(row.get("judge_success")):
            continue
        if success == "failed" and _truthy(row.get("judge_success")):
            continue
        if search:
            hay = " ".join(str(row.get(key) or "") for key in ("run_id", "sample_id", "text", "dataset_name", "benchmark_tag", "model_adapter", "attack"))
            if search.lower() not in hay.lower():
                continue
        filtered.append(row)
    artifact_order = {"complete": 3, "partial": 2, "summary_only": 1}

    # 中文注释：处理 sort_value 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
    def sort_value(item: dict[str, object]) -> tuple[int, object, str]:
        key = sort_by or "created"
        if key == "created":
            return (0, str(item.get("created_at") or item.get("run_id") or ""), str(item.get("sample_id") or ""))
        if key == "sample_id":
            return (0, f"{item.get('sample_id') or ''} {item.get('run_id') or ''}", str(item.get("created_at") or ""))
        if key == "task_dataset":
            return (0, f"{item.get('task_kind') or ''} {item.get('dataset_name') or ''} {item.get('benchmark_tag') or ''}", str(item.get("sample_id") or ""))
        if key == "model_attack":
            return (0, f"{item.get('model_adapter') or ''} {item.get('attack') or ''}", str(item.get("sample_id") or ""))
        if key == "status":
            return (0, 1 if _truthy(item.get("judge_success")) else 0, _to_float(item.get("risk_score"), 0.0))
        if key == "artifact":
            return (0, artifact_order.get(str(item.get("artifact_status") or ""), 0), str(item.get("evidence_confidence") or ""))
        if key == "report":
            return (0, str(item.get("run_id") or ""), str(item.get("sample_id") or ""))
        return (1, str(item.get("created_at") or item.get("run_id") or ""), str(item.get("sample_id") or ""))

    filtered.sort(key=sort_value, reverse=(sort_dir != "asc"))
    total, items = paginate(filtered, page=page, page_size=page_size)
    return RowsResponse(total=total, page=page, page_size=page_size, items=items)


# 中文注释：处理 compare_runs 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/compare", response_model=RunCompareResponse)
def compare_runs(
    request: Request,
    run_ids: str = Query(default=""),
):
    ids = [x.strip() for x in str(run_ids).split(",") if x.strip()]
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="run_ids must include at least two ids")

    art_dir = _artifacts_dir(request)
    per_run: dict[str, dict[str, object]] = {}
    per_victim: dict[str, dict[str, object]] = {}

    for rid in ids:
        if _is_generated_only_run({"run_id": rid}, art_dir) or _is_fake_model_run({"run_id": rid}, art_dir):
            continue
        run_root = _run_dir(rid, art_dir)
        summary = read_json(run_root / "summary.json", {})
        report = read_json(run_root / "report_data.json", {})
        if not isinstance(summary, dict):
            continue
        summary = apply_compatible_risk(summary)
        report = apply_compatible_report_data(report if isinstance(report, dict) else {})
        victims = summary.get("victims", {}) if isinstance(summary.get("victims"), dict) else {}
        per_run[rid] = {
            "summary": _without_retired_result_fields(summary),
            "stage_metrics": report.get("stage_metrics", {}),
        }

        for vname, payload in victims.items():
            node = payload if isinstance(payload, dict) else {}
            clean = node.get("clean", {}) if isinstance(node.get("clean"), dict) else {}
            attacked = node.get("attacked", {}) if isinstance(node.get("attacked"), dict) else {}
            clean_recall = float(clean.get("ir_r@1", 0.0))
            attacked_recall = float(attacked.get("ir_r@1", 0.0))
            attack_drop = clean_recall - attacked_recall
            slot = per_victim.setdefault(str(vname), {"runs": {}})
            runs = slot.setdefault("runs", {})
            if isinstance(runs, dict):
                runs[rid] = {
                    "attack_drop": attack_drop,
                    "clean_recall": clean_recall,
                    "attacked_recall": attacked_recall,
                }

    return RunCompareResponse(
        run_ids=ids,
        compare={
            "runs": per_run,
            "victims": per_victim,
        },
    )


# 中文注释：处理 get_summary 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{run_id}/summary")
def get_summary(run_id: str, request: Request):
    artifacts_dir = _artifacts_dir(request)
    _raise_if_unservable_run(run_id, artifacts_dir)
    p = _run_dir(run_id, artifacts_dir) / "summary.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="summary not found")
    return _without_retired_result_fields(apply_compatible_risk(read_json(p, {})))


# 中文注释：处理 get_results 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{run_id}/results", response_model=RowsResponse)
def get_results(
    run_id: str,
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
):
    artifacts_dir = _artifacts_dir(request)
    _raise_if_unservable_run(run_id, artifacts_dir)
    p = _run_dir(run_id, artifacts_dir) / "results.jsonl"
    rows = read_jsonl(p)
    rows = [_without_retired_result_fields(row) for row in rows]
    total, items = paginate(rows, page=page, page_size=page_size)
    return RowsResponse(total=total, page=page, page_size=page_size, items=items)


# 中文注释：处理 get_report_data 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{run_id}/report-data")
def get_report_data(run_id: str, request: Request):
    artifacts_dir = _artifacts_dir(request)
    _raise_if_unservable_run(run_id, artifacts_dir)
    p = _run_dir(run_id, artifacts_dir) / "report_data.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="report_data not found")
    return _without_retired_result_fields(apply_compatible_report_data(read_json(p, {})))


# 中文注释：处理 get_cases 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{run_id}/cases", response_model=RowsResponse)
def get_cases(
    run_id: str,
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
):
    artifacts_dir = _artifacts_dir(request)
    _raise_if_unservable_run(run_id, artifacts_dir)
    p = _run_dir(run_id, artifacts_dir) / "cases_index.jsonl"
    rows = read_jsonl(p)
    if not rows:
        rows = _derived_vlr_case_rows(run_id, artifacts_dir, limit=500)
    summary_risk = derive_compatible_risk(_summary_for_run(run_id, artifacts_dir))
    for row in rows:
        row["risk_level"] = summary_risk.get("risk_level", "")
        row["risk_score"] = summary_risk.get("risk_score", 0.0)
    total, items = paginate(rows, page=page, page_size=page_size)
    return RowsResponse(total=total, page=page, page_size=page_size, items=items)


# 中文注释：处理 get_case_detail 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{run_id}/cases/{sample_id}", response_model=CaseDetailResponse)
def get_case_detail(run_id: str, sample_id: str, request: Request) -> CaseDetailResponse:
    artifacts_dir = _artifacts_dir(request)
    _raise_if_unservable_run(run_id, artifacts_dir)
    run_root = _run_dir(run_id, artifacts_dir)
    case_dir = run_root / "cases" / sample_id
    bundle = read_json(case_dir / "case_bundle.json", {})
    if not bundle:
        derived = _derived_vlr_case_bundle(run_id, sample_id, artifacts_dir)
        if derived is None:
            raise HTTPException(status_code=404, detail="case not found")
        derived_bundle, derived_debug = derived
        return CaseDetailResponse(run_id=run_id, sample_id=sample_id, case_bundle=_without_retired_result_fields(derived_bundle), attack_debug=derived_debug)
    summary = _summary_for_run(run_id, artifacts_dir)
    bundle = _enrich_case_bundle_inputs(bundle, run_root, sample_id)
    bundle = _enrich_case_bundle_visual_refs(bundle, run_root, sample_id)

    debug_dirs = _case_debug_dirs(run_root, bundle, sample_id)
    debug_root = run_root / "attack_debug"
    debug_files: list[str] = []
    for debug_dir in debug_dirs:
        for item in sorted(debug_dir.glob("*")):
            if not item.is_file():
                continue
            try:
                debug_files.append(item.relative_to(debug_root).as_posix())
            except ValueError:
                debug_files.append(item.name)
    debug_payload: dict[str, object] = {
        "files": debug_files,
    }
    for debug_dir in debug_dirs:
        for name in ("debug.json", "advedm_plus_debug.json", "advedm_debug.json", "classic_attack_debug.json"):
            dbg_json = debug_dir / name
            if dbg_json.exists():
                debug_payload["debug"] = read_json(dbg_json, {})
                break
        if "debug" in debug_payload:
            break

    bundle = _attach_artifact_capability(bundle, run_root, summary, debug_files)
    bundle = _without_retired_result_fields(bundle)
    return CaseDetailResponse(run_id=run_id, sample_id=sample_id, case_bundle=bundle, attack_debug=debug_payload)


# 中文注释：处理 get_run_asset 对应的接口请求，并把后端接口路由结果整理为前端可消费的数据。
@router.get("/{run_id}/assets/{asset_path:path}")
def get_run_asset(run_id: str, asset_path: str, request: Request):
    artifacts_dir = _artifacts_dir(request)
    _raise_if_unservable_run(run_id, artifacts_dir)
    base = _run_dir(run_id, artifacts_dir)
    resolved_base = base.resolve()
    target = (base / asset_path).resolve()
    try:
        target.relative_to(resolved_base)
    except ValueError:
        raise HTTPException(status_code=403, detail="invalid asset path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(str(target))
