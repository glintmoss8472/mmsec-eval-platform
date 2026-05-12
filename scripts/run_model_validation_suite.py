# 文件说明：该文件属于运维与实验脚本，集中实现 run model validation suite 相关逻辑。
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_ADAPTERS, local_vlm_spec_by_adapter
from verify_live_server import ApiClient, MAIN_MODELS


UTC = timezone.utc
DEFAULT_ATTACKS = ("fgsm", "advedm_plus")
MIN_ATTACK_ASR_ANY = 0.02
MIN_ATTACK_DROP_R1_ANY = 0.02
MIN_QUALIFYING_ATTACKS_PER_MODEL = 1
MIN_CLEAN_R1_MEAN = 0.25
LOCAL_OPENAI_ADAPTERS = set(LOCAL_OPENAI_COMPAT_ADAPTERS)


# 执行 `now iso` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# 推断 `数据集 override`，从样本、配置或运行记录中提取统一名称。
def _dataset_override(dataset_name: str) -> dict[str, Any]:
    if dataset_name != "flickr1k":
        raise KeyError(f"unsupported validation dataset: {dataset_name}")
    return {
        "kind": "flickr1k",
        "root": "data/flickr30k",
        "image_dir": "images",
        "captions_file": "captions_index_single.jsonl",
        "split": "test",
        "max_items": 256,
        "benchmark_tag": "validation_flickr1k_256",
    }


# 推断 `攻击 作用范围`，从样本、配置或运行记录中提取统一名称。
def _attack_scope(attack: str) -> str:
    return "joint" if attack in {"tmm", "advedm_plus"} else "image"


# 推断 `攻击 params`，从样本、配置或运行记录中提取统一名称。
def _attack_params(attack: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "epsilon": 0.05,
        "step_size": 0.01,
        "steps": 8,
        "patch_size": 16,
        "topk": 6,
        "eps_t": 1,
        "text_candidates_k": 10,
        "patch_train_steps": 120,
    }
    if attack == "fgsm":
        common["steps"] = 1
        common["step_size"] = common["epsilon"]
    elif attack == "advedm_plus":
        common["steps"] = 12
    return common


# 推断 `配置 所属 攻击`，从样本、配置或运行记录中提取统一名称。
def _config_for_attack(attack: str) -> str:
    if attack == "advedm_plus":
        return "configs/bench/bootstrap_full_vlr_advedm_plus_cuda.yaml"
    return "configs/bench/bootstrap_full_vlr_cuda.yaml"


# 组装 `载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _payload(
    *,
    dataset_name: str,
    attack: str,
    model_adapter: str,
    experiment_id: str,
    seed: int,
    runtime_device: str,
    max_pairs: int,
    openai_timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "job_type": "run_vlr",
        "config_path": _config_for_attack(attack),
        "override": {
            "seed": int(seed),
            "runtime": {"device": str(runtime_device)},
            "model": {
                "openai_timeout": int(openai_timeout_seconds),
                "http_timeout": int(openai_timeout_seconds),
            },
            "plugins": {"attack": attack, "model_adapter": "clip_hf"},
            "dataset": {**_dataset_override(dataset_name), "benchmark_tag": f"{experiment_id}_{dataset_name}"},
            "task": {
                "kind": "vlr",
                "eval_scope": _attack_scope(attack),
                "compare_stages": ["clean", "attacked", "defended"],
            },
            "runner": {
                "surrogate_model_adapter": "clip_hf",
                "victim_model_adapters": [model_adapter],
                "max_pairs": int(max_pairs),
                "experiment_id": experiment_id,
                "save_plots": False,
            },
            "report": {"save_heatmaps": True, "save_patch_preview": True, "top_k_cases": 6},
            "sample_store": {"enabled": True, "save_images": True, "save_traces": True},
            "defense": {
                "enabled": True,
                "apply_on_clean": True,
                "apply_on_attacked": True,
            },
            "attack": _attack_params(attack),
        },
        "benchmark_mode": False,
    }


# 执行 `canonical 实验 id` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _canonical_experiment_id(model_adapter: str, attack: str) -> str:
    return f"scientific_validation_{str(model_adapter)}_{str(attack)}"


# 整理 `行记录 实验 id` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _row_experiment_id(row: dict[str, Any]) -> str:
    return str(row.get("experiment_id", "") or "").strip()


# 整理 `行记录 identity` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _row_identity(row: dict[str, Any]) -> tuple[str, ...]:
    experiment_id = _row_experiment_id(row)
    if experiment_id:
        return ("experiment_id", experiment_id)
    return ("model_attack", str(row.get("model_adapter", "")), str(row.get("attack", "")))


# 判断 `是否 primary 验证 行记录` 条件是否成立，为调用方提供布尔决策。
def _is_primary_validation_row(row: dict[str, Any], *, dataset_name: str, attacks: list[str]) -> bool:
    model_adapter = str(row.get("model_adapter", "") or "").strip()
    attack = str(row.get("attack", "") or "").strip()
    if not model_adapter or not attack:
        return False
    if str(row.get("dataset_name", "") or "").strip() != str(dataset_name):
        return False
    if attack not in {str(item) for item in attacks}:
        return False
    return _row_experiment_id(row) == _canonical_experiment_id(model_adapter, attack)


# 整理 `行记录 key` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("model_adapter", "")), str(row.get("attack", "")))


# 标记 `r1` 阶段，区分 clean、attacked 和 defended 样本。
def _stage_r1(metrics: dict[str, Any]) -> float:
    return 0.5 * (
        float(metrics.get("ir_r@1", 0.0) or 0.0)
        + float(metrics.get("tr_r@1", 0.0) or 0.0)
    )


# 提取 `运行记录 证据`，从归档、结果或响应中取出后续流程需要的字段。
def _extract_run_evidence(summary: dict[str, Any]) -> dict[str, float]:
    victim_compare = [item for item in list(summary.get("victim_compare", [])) if isinstance(item, dict)]
    defense_compare = [item for item in list(summary.get("defense_compare", [])) if isinstance(item, dict)]
    clean_r1_values = [_stage_r1(dict(item.get("clean", {}) or {})) for item in victim_compare]
    attacked_r1_values = [_stage_r1(dict(item.get("attacked", {}) or {})) for item in victim_compare]
    attack_drop_r1_values = [
        float(clean_r1) - float(attacked_r1)
        for clean_r1, attacked_r1 in zip(clean_r1_values, attacked_r1_values)
    ]
    defense_recovery_r1_values = [
        float(item.get("defense_recovery_r1", 0.0) or 0.0)
        for item in defense_compare
    ]
    defense_utility_drop_r1_values = [
        float(item.get("defense_utility_drop@1", 0.0) or 0.0)
        for item in defense_compare
    ]
    mean_rank_delta_values = [
        0.5
        * (
            float(item.get("delta_mean_rank_ir", 0.0) or 0.0)
            + float(item.get("delta_mean_rank_tr", 0.0) or 0.0)
        )
        for item in victim_compare
    ]
    return {
        "clean_r1_mean": mean(clean_r1_values) if clean_r1_values else 0.0,
        "attacked_r1_mean": mean(attacked_r1_values) if attacked_r1_values else 0.0,
        "attack_drop_r1_mean": mean(attack_drop_r1_values) if attack_drop_r1_values else 0.0,
        "defense_recovery_r1_mean": mean(defense_recovery_r1_values) if defense_recovery_r1_values else 0.0,
        "defense_utility_drop_r1_mean": mean(defense_utility_drop_r1_values) if defense_utility_drop_r1_values else 0.0,
        "mean_rank_delta_mean": mean(mean_rank_delta_values) if mean_rank_delta_values else 0.0,
    }


# 整理 `hydrate 行记录 来源 运行记录 摘要` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _hydrate_row_from_run_summary(row: dict[str, Any], *, project_root: Path | None = None) -> dict[str, Any]:
    if str(row.get("job_status", "")).strip().lower() != "success":
        return row
    if row.get("attack_drop_r1_mean") is not None:
        return row
    run_id = str(row.get("run_id", "")).strip()
    if not run_id:
        return row
    root = (project_root or Path.cwd()).resolve()
    summary_path = root / "artifacts" / "runs" / run_id / "summary.json"
    if not summary_path.exists():
        return row
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return row
    row.update(_extract_run_evidence(summary))
    return row


# 整理 `dedupe rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    index_by_key: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = _row_identity(row)
        if key in index_by_key:
            deduped[index_by_key[key]] = row
        else:
            index_by_key[key] = len(deduped)
            deduped.append(row)
    return deduped


# 加载 `已有 rows`，把外部文件、配置或运行产物转换为内存结构。
def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = _dedupe_rows(list(data.get("rows", [])))
    for row in rows:
        _hydrate_row_from_run_summary(row)
    return rows


# 执行 `successful keys` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _successful_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for row in rows:
        if str(row.get("job_status", "")).strip().lower() != "success":
            continue
        key = _row_key(row)
        if _row_experiment_id(row) != _canonical_experiment_id(key[0], key[1]):
            continue
        out.add(key)
    return out


# 整理 `upsert 行记录` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _upsert_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = _row_identity(row)
    for index, existing in enumerate(rows):
        if _row_identity(existing) == key:
            rows[index] = row
            return
    rows.append(row)


# 整理 `模型 摘要 行记录` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _model_summary_row(
    *,
    model_adapter: str,
    items: list[dict[str, Any]],
    attacks: list[str],
    minimum_qualifying_attack_count_per_model: int,
) -> tuple[dict[str, Any], bool, bool]:
    success_items = [item for item in items if str(item.get("job_status", "")) == "success"]
    covered = sorted({str(item.get("attack", "")) for item in success_items})
    failures = sum(int(item.get("num_victim_failures", 0) or 0) for item in success_items)
    asr_values = [float(item.get("asr_attack", 0.0) or 0.0) for item in success_items]
    defense_values = [float(item.get("defense_gain", 0.0) or 0.0) for item in success_items]
    clean_r1_values = [float(item.get("clean_r1_mean", 0.0) or 0.0) for item in success_items]
    attack_drop_values = [float(item.get("attack_drop_r1_mean", 0.0) or 0.0) for item in success_items]
    defense_recovery_values = [float(item.get("defense_recovery_r1_mean", 0.0) or 0.0) for item in success_items]
    qualifying_items = [
        item
        for item in success_items
        if float(item.get("asr_attack", 0.0) or 0.0) >= float(MIN_ATTACK_ASR_ANY)
        or float(item.get("attack_drop_r1_mean", 0.0) or 0.0) >= float(MIN_ATTACK_DROP_R1_ANY)
    ]
    qualifying_attacks = sorted({str(item.get("attack", "")) for item in qualifying_items if str(item.get("attack", "")).strip()})
    evidence_ok = failures == 0 and bool(success_items)
    clean_baseline_ok = (mean(clean_r1_values) if clean_r1_values else 0.0) >= float(MIN_CLEAN_R1_MEAN)
    retrieval_drop_signal_ok = (max(attack_drop_values) if attack_drop_values else 0.0) >= float(MIN_ATTACK_DROP_R1_ANY)
    ok = evidence_ok and len(qualifying_attacks) >= int(minimum_qualifying_attack_count_per_model)
    scientific_quality_ok = ok and clean_baseline_ok and retrieval_drop_signal_ok
    return {
        "model_adapter": str(model_adapter),
        "benchmark_attacks": list(attacks),
        "covered_attacks": covered,
        "qualifying_attacks": qualifying_attacks,
        "success_count": len(success_items),
        "qualifying_attack_count": len(qualifying_attacks),
        "num_victim_failures": failures,
        "asr_attack_mean": mean(asr_values) if asr_values else 0.0,
        "asr_attack_max": max(asr_values) if asr_values else 0.0,
        "asr_defended_mean": mean([float(item.get("asr_defended", 0.0) or 0.0) for item in success_items]) if success_items else 0.0,
        "defense_gain_mean": mean(defense_values) if defense_values else 0.0,
        "defense_gain_max": max(defense_values) if defense_values else 0.0,
        "clean_r1_mean": mean(clean_r1_values) if clean_r1_values else 0.0,
        "attack_drop_r1_mean": mean(attack_drop_values) if attack_drop_values else 0.0,
        "attack_drop_r1_max": max(attack_drop_values) if attack_drop_values else 0.0,
        "defense_recovery_r1_mean": mean(defense_recovery_values) if defense_recovery_values else 0.0,
        "evidence_ok": evidence_ok,
        "scientific_signal_ok": bool(qualifying_items),
        "clean_baseline_ok": clean_baseline_ok,
        "retrieval_drop_signal_ok": retrieval_drop_signal_ok,
        "validated": ok,
        "scientific_quality_ok": scientific_quality_ok,
    }, ok, scientific_quality_ok


# 整理 `supplementary 行记录 preview` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _supplementary_row_preview(supplementary_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "experiment_id": _row_experiment_id(row),
            "model_adapter": str(row.get("model_adapter", "") or ""),
            "attack": str(row.get("attack", "") or ""),
            "job_status": str(row.get("job_status", "") or ""),
        }
        for row in supplementary_rows
    ]


# 执行 `summarize` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _summarize(
    rows: list[dict[str, Any]],
    *,
    attacks: list[str],
    dataset_name: str,
    required_model_count: int,
    max_pairs: int,
    minimum_qualifying_attack_count_per_model: int,
    model_adapters: list[str] | None = None,
) -> dict[str, Any]:
    primary_rows = [row for row in rows if _is_primary_validation_row(row, dataset_name=dataset_name, attacks=attacks)]
    supplementary_rows = [row for row in rows if row not in primary_rows]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_rows:
        grouped[str(row.get("model_adapter", ""))].append(row)

    per_model: list[dict[str, Any]] = []
    validated_models: list[str] = []
    scientific_quality_validated_models: list[str] = []
    missing_models: list[str] = []
    global_success_attacks: set[str] = set()
    summary_models = list(model_adapters or MAIN_MODELS)
    for model_adapter in summary_models:
        items = grouped.get(str(model_adapter), [])
        success_items = [item for item in items if str(item.get("job_status", "")) == "success"]
        global_success_attacks.update(str(item.get("attack", "")) for item in success_items if str(item.get("attack", "")).strip())
        row, ok, scientific_quality_ok = _model_summary_row(
            model_adapter=str(model_adapter),
            items=items,
            attacks=attacks,
            minimum_qualifying_attack_count_per_model=minimum_qualifying_attack_count_per_model,
        )
        per_model.append(row)
        if ok:
            validated_models.append(str(model_adapter))
        if scientific_quality_ok:
            scientific_quality_validated_models.append(str(model_adapter))
        else:
            missing_models.append(str(model_adapter))

    benchmark_attack_coverage_ok = {str(item) for item in attacks}.issubset(global_success_attacks)
    return {
        "generated_at": _now_iso(),
        "dataset_name": str(dataset_name),
        "attacks": list(attacks),
        "benchmark_attack_coverage_ok": benchmark_attack_coverage_ok,
        "required_model_count": int(required_model_count),
        "primary_row_count": len(primary_rows),
        "supplementary_row_count": len(supplementary_rows),
        "successful_row_count": sum(1 for row in primary_rows if str(row.get("job_status", "")) == "success"),
        "failed_row_count": sum(1 for row in primary_rows if str(row.get("job_status", "")) != "success"),
        "validated_models": validated_models,
        "validated_model_count": len(validated_models),
        "scientific_quality_validated_models": scientific_quality_validated_models,
        "scientific_quality_model_count": len(scientific_quality_validated_models),
        "scientific_quality_passed": len(scientific_quality_validated_models) >= int(required_model_count) and benchmark_attack_coverage_ok,
        "missing_models": missing_models,
        "supplementary_rows": _supplementary_row_preview(supplementary_rows),
        "per_model": per_model,
        "rows": rows,
        "passed": len(validated_models) >= int(required_model_count) and benchmark_attack_coverage_ok,
        "criterion": {
            "dataset_name": str(dataset_name),
            "benchmark_attacks": list(attacks),
            "max_pairs": int(max_pairs),
            "require_zero_victim_failures": True,
            "minimum_qualifying_attack_count_per_model": int(minimum_qualifying_attack_count_per_model),
            "minimum_attack_asr_any": float(MIN_ATTACK_ASR_ANY),
            "minimum_attack_drop_r1_any": float(MIN_ATTACK_DROP_R1_ANY),
            "minimum_clean_r1_mean": float(MIN_CLEAN_R1_MEAN),
            "description": "lightweight transfer validation on the Flickr1k split with a fixed pair budget, recorded clean/attack/defense evidence, and at least one non-trivial adversarial effect per validated model across the benchmark attack set",
        },
    }


# 解析 `args`，把文本或载荷转换成可校验的字段。
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--dataset", default="flickr1k")
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--models", default="")
    parser.add_argument("--seed-base", type=int, default=20260418)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--runtime-device", default=os.getenv("MMSEC_VALIDATION_RUNTIME_DEVICE", "cuda:0"))
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--min-qualifying-attacks-per-model", type=int, default=MIN_QUALIFYING_ATTACKS_PER_MODEL)
    parser.add_argument("--openai-timeout-seconds", type=int, default=180)
    parser.add_argument("--local-vlm-startup-timeout-seconds", type=int, default=900)
    parser.add_argument("--no-auto-launch-local-vlm", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


# 整理 `验证 paths` 路径信息，把本地文件或产物引用转换成统一表示。
def _validation_paths(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path("artifacts") / f"model_validation_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "rows_path": out_dir / "rows.json",
        "status_path": out_dir / "status.json",
        "summary_path": out_dir / "summary.json",
    }


# 执行 `验证 targets` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _validation_targets(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    attacks = [item.strip() for item in str(args.attacks or "").split(",") if item.strip()]
    selected_models = {item.strip() for item in str(args.models or "").split(",") if item.strip()}
    model_adapters = [item for item in MAIN_MODELS if not selected_models or item in selected_models]
    return attacks, model_adapters


# 拼接 `本地 视觉语言模型 模型 URL`，把配置中的主机、端口和路径合成实际访问入口。
def _local_vlm_models_url(model_adapter: str) -> str:
    spec = local_vlm_spec_by_adapter(model_adapter)
    return f"{spec.endpoint_default.rstrip('/')}/models"


# 判断或归一 `本地 视觉语言模型 ready` 状态，让调用方可以稳定渲染能力和可用性。
def _local_vlm_ready(model_adapter: str, *, timeout_seconds: float = 5.0) -> bool:
    if model_adapter not in LOCAL_OPENAI_ADAPTERS:
        return True
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(_local_vlm_models_url(model_adapter), timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException:
        return False
    return True


# 判断或归一 `process alive` 状态，让调用方可以稳定渲染能力和可用性。
def _process_alive(pid: str) -> bool:
    if not str(pid or "").strip().isdigit():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# 执行 `tail 文本` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _tail_text(path: Path, max_chars: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:]
    except OSError:
        return ""


# 推断 `drop 本地 模型 file cache`，从样本、配置或运行记录中提取统一名称。
def _drop_local_model_file_cache() -> dict[str, Any]:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return {"enabled": False, "reason": "posix_fadvise unavailable"}

    advised_files = 0
    advised_bytes = 0
    for pattern in ("artifacts/local_vlm/**/*.safetensors", "artifacts/local_vlm/**/*.bin"):
        for path in PROJECT_ROOT.glob(pattern):
            try:
                if not path.is_file() or path.stat().st_size < 1024 * 1024:
                    continue
                fd = os.open(str(path), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                advised_files += 1
                advised_bytes += int(path.stat().st_size)
            except OSError:
                continue

    return {
        "enabled": True,
        "advised_files": advised_files,
        "advised_bytes": advised_bytes,
    }


# 执行 `wait 所属 本地 视觉语言模型` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _wait_for_local_vlm(
    model_adapter: str,
    *,
    timeout_seconds: int,
    poll_seconds: float,
    server_pid: str = "",
    launch_log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    while time.monotonic() < deadline:
        if _local_vlm_ready(model_adapter, timeout_seconds=min(10.0, max(1.0, float(poll_seconds)))):
            return
        if server_pid and not _process_alive(server_pid):
            log_tail = _tail_text(launch_log_path) if launch_log_path is not None else ""
            raise RuntimeError(
                f"local VLM server exited before ready: {model_adapter}; "
                f"pid={server_pid}; log_tail={log_tail}"
            )
        time.sleep(max(1.0, float(poll_seconds)))
    raise TimeoutError(f"local VLM server did not become ready: {model_adapter}")


# 停止 `本地 视觉语言模型 servers`，释放端口、进程或显存等运行资源。
def _stop_local_vlm_servers() -> dict[str, Any]:
    result = subprocess.run(
        ["pkill", "-f", "local_openai_mm_server.py"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return {
        "stopped": result.returncode in {0, 1},
        "returncode": int(result.returncode),
        "stderr": result.stderr[-1000:],
    }


# 执行 `launch 本地 视觉语言模型` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _launch_local_vlm(model_adapter: str, *, startup_timeout_seconds: int, poll_seconds: float) -> dict[str, Any]:
    if model_adapter not in LOCAL_OPENAI_ADAPTERS:
        cleanup = _stop_local_vlm_servers()
        return {"adapter": model_adapter, "launched": False, "reason": "classic_adapter", "cleanup": cleanup}
    if _local_vlm_ready(model_adapter):
        return {
            "adapter": model_adapter,
            "launched": False,
            "reason": "already_ready",
            "models_url": _local_vlm_models_url(model_adapter),
        }

    spec = local_vlm_spec_by_adapter(model_adapter)
    script_path = PROJECT_ROOT / spec.launch_script
    if not script_path.exists():
        raise FileNotFoundError(f"missing local VLM launch script: {script_path}")

    cache_reclaim = _drop_local_model_file_cache()
    env = dict(os.environ)
    env.setdefault("MMSEC_LOCAL_VLM_REQUIRE_OFFLINE", "1")
    env.setdefault("MMSEC_LOCAL_VLM_SINGLE_TENANT", "auto")
    env.setdefault("MMSEC_MODEL_SERVER_PREFLIGHT", "0")
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "local VLM launch failed: "
            f"{model_adapter}\nstdout={result.stdout[-1000:]}\nstderr={result.stderr[-2000:]}"
        )

    server_pid = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    launch_log_path = PROJECT_ROOT / "logs" / "model_servers" / spec.launch_log
    _wait_for_local_vlm(
        model_adapter,
        timeout_seconds=int(startup_timeout_seconds),
        poll_seconds=float(poll_seconds),
        server_pid=server_pid,
        launch_log_path=launch_log_path,
    )
    return {
        "adapter": model_adapter,
        "launched": True,
        "script": str(script_path),
        "pid": server_pid,
        "models_url": _local_vlm_models_url(model_adapter),
        "cache_reclaim": cache_reclaim,
    }


# 判断或归一 `initial 验证 状态` 状态，让调用方可以稳定渲染能力和可用性。
def _initial_validation_status(args: argparse.Namespace, attacks: list[str], model_adapters: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "status": "running",
        "dataset_name": str(args.dataset),
        "attacks": attacks,
        "models": model_adapters,
        "runtime_device": str(args.runtime_device),
        "max_pairs": int(args.max_pairs),
        "auto_launch_local_vlm": not bool(args.no_auto_launch_local_vlm),
        "completed": len(rows),
        "expected_total": len(model_adapters) * len(attacks),
    }


# 整理 `new 验证 行记录` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _new_validation_row(args: argparse.Namespace, model_adapter: str, attack: str, model_index: int, attack_index: int, attacks: list[str]) -> dict[str, Any]:
    return {
        "submitted_at": _now_iso(),
        "dataset_name": str(args.dataset),
        "attack": str(attack),
        "model_adapter": str(model_adapter),
        "seed": int(args.seed_base) + model_index * max(1, len(attacks)) + attack_index,
        "experiment_id": f"scientific_validation_{model_adapter}_{attack}",
    }


# 执行 `submit 验证 任务` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _submit_validation_job(args: argparse.Namespace, api: ApiClient, row: dict[str, Any]) -> None:
    created = api.create_job(
        _payload(
            dataset_name=str(args.dataset),
            attack=str(row["attack"]),
            model_adapter=str(row["model_adapter"]),
            experiment_id=str(row["experiment_id"]),
            seed=int(row["seed"]),
            runtime_device=str(args.runtime_device),
            max_pairs=int(args.max_pairs),
            openai_timeout_seconds=int(args.openai_timeout_seconds),
        )
    )
    row["job_id"] = str(created.get("id", ""))


# 执行 `complete 验证 任务` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _complete_validation_job(args: argparse.Namespace, api: ApiClient, row: dict[str, Any]) -> None:
    final_job = _wait_for_validation_job(
        api,
        row["job_id"],
        timeout_seconds=int(args.timeout_seconds),
        poll_seconds=float(args.poll_seconds),
    )
    row["job_status"] = str(final_job.get("status", ""))
    row["finished_at"] = _now_iso()
    if row["job_status"] != "success":
        logs = api.get_job_logs(row["job_id"], page_size=200)
        row["logs_tail"] = logs.get("items", [])[-20:]
        return
    _clear_failure_fields(row)
    run_id = str(final_job.get("run_id", ""))
    row["run_id"] = run_id
    summary = api.get_run_summary(run_id)
    row["asr_attack"] = float(summary.get("asr_attack", summary.get("asr", 0.0)) or 0.0)
    row["asr_defended"] = float(summary.get("asr_defended", 0.0) or 0.0)
    row["defense_gain"] = float(summary.get("defense_gain", 0.0) or 0.0)
    row["risk_score"] = float(summary.get("risk_score", 0.0) or 0.0)
    row["num_victim_failures"] = int(summary.get("num_victim_failures", 0) or 0)
    row.update(_extract_run_evidence(summary))


# 执行 `mark 验证 任务 failed` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _mark_validation_job_failed(api: ApiClient, row: dict[str, Any], exc: Exception) -> None:
    row["job_status"] = "failed"
    row["finished_at"] = _now_iso()
    row["exception"] = str(exc)
    row["traceback"] = traceback.format_exc()
    try:
        logs = api.get_job_logs(row["job_id"], page_size=200)
        row["logs_tail"] = logs.get("items", [])[-20:]
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, requests.RequestException) as log_exc:
        row["logs_error"] = str(log_exc)


# 执行 `clear failure fields` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _clear_failure_fields(row: dict[str, Any]) -> None:
    for key in ("exception", "traceback", "logs_error"):
        row.pop(key, None)


# 执行 `one 验证 任务` 流程，按配置驱动运维与实验脚本完成一次任务。
def _run_one_validation_job(args: argparse.Namespace, api: ApiClient, row: dict[str, Any]) -> dict[str, Any]:
    _submit_validation_job(args, api, row)
    try:
        _complete_validation_job(args, api, row)
    except (KeyError, OSError, RuntimeError, TimeoutError, TypeError, ValueError, requests.RequestException) as exc:
        _mark_validation_job_failed(api, row, exc)
    return row


# 执行 `recover previous 验证 任务` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _recover_previous_validation_jobs(args: argparse.Namespace, api: ApiClient, rows: list[dict[str, Any]]) -> None:
    """Attach to unfinished jobs from a previous resume run before submitting duplicates."""
    for row in rows:
        if str(row.get("job_status", "")).strip().lower() == "success":
            continue
        job_id = str(row.get("job_id", "") or "").strip()
        if not job_id:
            continue
        try:
            job = api.get_job(job_id)
        except requests.RequestException as exc:
            row["recovery_error"] = str(exc)
            continue
        job_status = str(job.get("status", "") or "").strip().lower()
        if job_status in {"queued", "running"}:
            row["recovered_from_previous_job"] = True
            row["previous_failed_status"] = row.get("job_status", "")
            row["job_status"] = job_status
            try:
                _complete_validation_job(args, api, row)
            except (KeyError, OSError, RuntimeError, TimeoutError, TypeError, ValueError, requests.RequestException) as exc:
                _mark_validation_job_failed(api, row, exc)
        elif job_status == "success":
            row["recovered_from_previous_job"] = True
            row["previous_failed_status"] = row.get("job_status", "")
            try:
                row["job_status"] = "success"
                row["finished_at"] = _now_iso()
                _clear_failure_fields(row)
                run_id = str(job.get("run_id", "") or "")
                row["run_id"] = run_id
                summary = api.get_run_summary(run_id)
                row["asr_attack"] = float(summary.get("asr_attack", summary.get("asr", 0.0)) or 0.0)
                row["asr_defended"] = float(summary.get("asr_defended", 0.0) or 0.0)
                row["defense_gain"] = float(summary.get("defense_gain", 0.0) or 0.0)
                row["risk_score"] = float(summary.get("risk_score", 0.0) or 0.0)
                row["num_victim_failures"] = int(summary.get("num_victim_failures", 0) or 0)
                row.update(_extract_run_evidence(summary))
            except (KeyError, OSError, RuntimeError, TypeError, ValueError, requests.RequestException) as exc:
                _mark_validation_job_failed(api, row, exc)
        else:
            row["observed_previous_job_status"] = job_status


# 执行 `wait 所属 验证 任务` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _wait_for_validation_job(api: ApiClient, job_id: str, *, timeout_seconds: int, poll_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(1, int(timeout_seconds))
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            job = api.get_job(job_id)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(max(1.0, float(poll_seconds)))
            continue
        status = str(job.get("status", ""))
        if status in {"success", "failed", "cancelled"}:
            return job
        time.sleep(max(1.0, float(poll_seconds)))
    if last_error is not None:
        raise TimeoutError(f"job timed out after transient API errors: {job_id}; last_error={last_error}") from last_error
    raise TimeoutError(f"job timed out: {job_id}")


# 整理 `persist 验证 行记录` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _persist_validation_row(rows: list[dict[str, Any]], row: dict[str, Any], rows_path: Path, status_path: Path, status: dict[str, Any]) -> None:
    _upsert_row(rows, row)
    rows_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    status["completed"] = len(rows)
    status["last_row"] = {
        "model_adapter": row["model_adapter"],
        "attack": row["attack"],
        "job_status": row["job_status"],
        "run_id": row.get("run_id", ""),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


# 执行 `验证 任务` 流程，按配置驱动运维与实验脚本完成一次任务。
def _run_validation_jobs(args: argparse.Namespace, api: ApiClient, attacks: list[str], model_adapters: list[str], rows: list[dict[str, Any]], rows_path: Path, status_path: Path, status: dict[str, Any]) -> None:
    done = _successful_keys(rows)
    for model_index, model_adapter in enumerate(model_adapters):
        pending_attacks = [
            str(attack)
            for attack in attacks
            if (str(model_adapter), str(attack)) not in done
        ]
        if not pending_attacks:
            status.setdefault("local_vlm_events", []).append(
                {
                    "adapter": str(model_adapter),
                    "launched": False,
                    "reason": "all_validation_rows_already_successful",
                    "ts": _now_iso(),
                }
            )
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        if not bool(args.no_auto_launch_local_vlm):
            launch_event = _launch_local_vlm(
                str(model_adapter),
                startup_timeout_seconds=int(args.local_vlm_startup_timeout_seconds),
                poll_seconds=float(args.poll_seconds),
            )
            status.setdefault("local_vlm_events", []).append({**launch_event, "ts": _now_iso()})
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        for attack_index, attack in enumerate(attacks):
            if str(attack) not in pending_attacks:
                continue
            row = _new_validation_row(args, model_adapter, attack, model_index, attack_index, attacks)
            row = _run_one_validation_job(args, api, row)
            _persist_validation_row(rows, row, rows_path, status_path, status)


# 写出 `验证 摘要`，保证后续报告、页面或复现实验能读取。
def _write_validation_summary(args: argparse.Namespace, rows: list[dict[str, Any]], attacks: list[str], model_adapters: list[str], summary_path: Path, status_path: Path, status: dict[str, Any]) -> int:
    summary = _summarize(
        rows,
        attacks=attacks,
        dataset_name=str(args.dataset),
        required_model_count=len(model_adapters),
        max_pairs=int(args.max_pairs),
        minimum_qualifying_attack_count_per_model=int(args.min_qualifying_attacks_per_model),
        model_adapters=model_adapters,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    status["status"] = "completed"
    status["ended_at"] = _now_iso()
    status["passed"] = bool(summary.get("passed", False))
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if bool(summary.get("passed", False)) else 1


# 执行 `mark 验证 suite failed` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _mark_validation_suite_failed(status: dict[str, Any], status_path: Path, exc: Exception) -> None:
    status["status"] = "failed"
    status["ended_at"] = _now_iso()
    status["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


# 作为 `run_model_validation_suite.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    args = _parse_args()
    paths = _validation_paths(args)
    attacks, model_adapters = _validation_targets(args)
    api = ApiClient(args.api_base, timeout=120.0)
    rows = _load_existing_rows(paths["rows_path"]) if args.resume else []
    status = _initial_validation_status(args, attacks, model_adapters, rows)
    status_path = paths["status_path"]
    rows_path = paths["rows_path"]
    summary_path = paths["summary_path"]
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if args.resume:
            _recover_previous_validation_jobs(args, api, rows)
            rows = _dedupe_rows(rows)
            rows_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
            status["completed"] = len(rows)
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        _run_validation_jobs(args, api, attacks, model_adapters, rows, rows_path, status_path, status)
        return _write_validation_summary(args, rows, attacks, model_adapters, summary_path, status_path, status)
    except (KeyError, OSError, RuntimeError, TimeoutError, TypeError, ValueError, requests.RequestException) as exc:
        _mark_validation_suite_failed(status, status_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
