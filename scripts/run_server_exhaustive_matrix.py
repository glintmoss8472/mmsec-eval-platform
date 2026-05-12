# 文件说明：该文件属于运维与实验脚本，集中实现 run server exhaustive matrix 相关逻辑。
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_paper_acceptance import evaluate_acceptance
from verify_live_server import ATTACK_SPECS, MAIN_MODELS, ApiClient, _wait_for_job


UTC = timezone.utc
TERMINAL_JOB_STATUSES = {"success", "failed", "cancelled"}
CLASSIC_MODELS = ("clip_hf", "blip_itm", "vilt_itm")
PAPER_ATTACKS = ("advclip", "tmm", "advedm", "advedm_plus")
THRESHOLD_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "paper_acceptance_thresholds.json"
PAPER_ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "server_snapshot_20260411" / "paper_suite_analysis.json"
DATASET_MAX_ITEMS_OVERRIDE = 0
MAX_PAIRS_OVERRIDE = -1


# 执行 `now iso` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
        "text_candidates_k": 8,
        "patch_train_steps": 80,
        "batch_size": 128,
        "cw_const": 0.1,
        "cw_confidence": 0.1,
    }
    if attack == "fgsm":
        common["steps"] = 1
        common["step_size"] = common["epsilon"]
    elif attack in {"bim", "pgd"}:
        common["steps"] = 10
        common["step_size"] = 0.01
    elif attack in {"mifgsm", "nifgsm", "difgsm", "tifgsm", "dtmifgsm", "vmifgsm", "vnifgsm"}:
        common["steps"] = 12
        common["step_size"] = 0.008
    elif attack == "cw":
        common["steps"] = 20
        common["step_size"] = 0.005
    elif attack == "advclip":
        common["steps"] = 12
        common["patch_train_steps"] = 160
    elif attack == "tmm":
        common["steps"] = 8
        common["text_candidates_k"] = 10
    elif attack == "advedm":
        common["steps"] = 12
    elif attack == "advedm_plus":
        common["steps"] = 12
        common["text_candidates_k"] = 10
    return common


# 执行 `with max items` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _with_max_items(payload: dict[str, Any], default_max_items: int, default_tag: str) -> dict[str, Any]:
    max_items = int(DATASET_MAX_ITEMS_OVERRIDE or default_max_items)
    payload["max_items"] = max_items
    payload["benchmark_tag"] = default_tag if max_items == int(default_max_items) else f"{default_tag}_n{max_items}"
    return payload


# 推断 `数据集 override`，从样本、配置或运行记录中提取统一名称。
def _dataset_override(dataset_name: str) -> dict[str, Any]:
    if dataset_name == "coco_subset":
        return _with_max_items(
            {
                "kind": "coco_subset",
                "root": "data/coco",
                "image_dir": "val2017",
                "captions_file": "annotations/captions_val2017_subset.json",
                "split": "val",
            },
            5000,
            "coco_subset_full5000",
        )
    if dataset_name == "flickr30k":
        return _with_max_items(
            {
                "kind": "flickr30k",
                "root": "data/flickr30k",
                "image_dir": "images",
                "captions_file": "captions_index.jsonl",
                "split": "test",
            },
            1280,
            "flickr30k_full1280",
        )
    if dataset_name == "flickr1k":
        return _with_max_items(
            {
                "kind": "flickr1k",
                "root": "data/flickr30k",
                "image_dir": "images",
                "captions_file": "captions_index_single.jsonl",
                "split": "test",
            },
            1000,
            "flickr1k_full1000",
        )
    if dataset_name == "mini_flickr":
        return _with_max_items(
            {
                "kind": "mini_flickr",
                "root": "",
                "image_dir": "images",
                "captions_file": "captions_index.jsonl",
                "split": "test",
            },
            16,
            "mini_flickr_full16",
        )
    raise KeyError(f"unsupported dataset: {dataset_name}")


# 执行 `max pairs 所属` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _max_pairs_for(dataset_name: str, model_adapter: str) -> int:
    if int(MAX_PAIRS_OVERRIDE) >= 0:
        return int(MAX_PAIRS_OVERRIDE)
    if dataset_name == "mini_flickr":
        return 256
    if model_adapter in {"clip_hf", "blip_itm", "vilt_itm"}:
        return 16384
    if dataset_name in {"flickr30k", "flickr1k"}:
        return 8192
    return 12288


# 执行 `mode override` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _mode_override(mode_name: str) -> tuple[list[str], dict[str, Any]]:
    if mode_name == "standard":
        return ["clean", "attacked"], {
            "enabled": False,
            "apply_on_clean": False,
            "apply_on_attacked": False,
        }
    if mode_name == "defense":
        return ["clean", "attacked", "defended"], {
            "enabled": True,
            "apply_on_clean": True,
            "apply_on_attacked": True,
        }
    raise KeyError(f"unsupported mode: {mode_name}")


# 组装 `任务 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _job_payload(
    *,
    attack: str,
    config_path: str,
    dataset_name: str,
    mode_name: str,
    model_adapter: str,
    repeat_index: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    compare_stages, defense = _mode_override(mode_name)
    benchmark_tag = f"suite_{dataset_name}_{mode_name}_{attack}_{model_adapter}_r{repeat_index}"
    experiment_id = benchmark_tag
    payload = {
        "job_type": "run_vlr",
        "config_path": config_path,
        "override": {
            "seed": int(seed),
            "plugins": {
                "attack": attack,
                "model_adapter": "clip_hf",
            },
            "dataset": {
                **_dataset_override(dataset_name),
                "benchmark_tag": benchmark_tag,
            },
            "task": {
                "kind": "vlr",
                "eval_scope": _attack_scope(attack),
                "compare_stages": compare_stages,
            },
            "runner": {
                "surrogate_model_adapter": "clip_hf",
                "victim_model_adapters": [model_adapter],
                "max_pairs": _max_pairs_for(dataset_name, model_adapter),
                "experiment_id": experiment_id,
                "save_plots": False,
            },
            # Full-matrix runs should prioritize numeric metrics.  Per-sample
            # debug images are generated in separate evidence runs, otherwise
            # every 5k run writes tens of thousands of tiny files and disables
            # batchable attacks.
            "runtime": {
                "num_workers": 8,
            },
            "report": {
                "save_heatmaps": False,
                "save_patch_preview": False,
                "top_k_cases": 8,
            },
            "sample_store": {
                "enabled": False,
                "save_images": False,
                "save_traces": False,
            },
            "defense": defense,
            "attack": _attack_params(attack),
        },
        "benchmark_mode": False,
    }
    return experiment_id, payload


# 整理 `行记录 key` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row.get("attack", "")),
        str(row.get("dataset_name", "")),
        str(row.get("mode_name", "")),
        str(row.get("model_adapter", "")),
        int(row.get("repeat_index", 0) or 0),
    )


# 加载 `已有 rows`，把外部文件、配置或运行产物转换为内存结构。
def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("rows", []))


# 加载 `thresholds`，把外部文件、配置或运行产物转换为内存结构。
def _load_thresholds() -> dict[str, Any]:
    if not THRESHOLD_PATH.exists():
        return {}
    return json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))


# 加载 `paper analysis`，把外部文件、配置或运行产物转换为内存结构。
def _load_paper_analysis() -> dict[str, Any]:
    if not PAPER_ANALYSIS_PATH.exists():
        return {}
    return json.loads(PAPER_ANALYSIS_PATH.read_text(encoding="utf-8"))


# 整理 `classic acceptance rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _classic_acceptance_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase_rows: dict[str, dict[str, dict[str, Any]]] = {}
    acceptance_rows: list[dict[str, Any]] = []
    for dataset_name, threshold_key, suite_name in (
        ("coco_subset", "e1", "E1_classic_coco"),
        ("flickr30k", "e2", "E2_classic_flickr"),
    ):
        phase_rows[threshold_key] = {}
        for attack in PAPER_ATTACKS:
            selected = [
                row
                for row in rows
                if str(row.get("dataset_name", "")) == dataset_name
                and str(row.get("attack", "")) == attack
                and str(row.get("mode_name", "")) == "defense"
                and str(row.get("model_adapter", "")) in CLASSIC_MODELS
                and str(row.get("job_status", "")) == "success"
            ]
            asr_values = [float(row.get("asr_attack", 0.0) or 0.0) for row in selected]
            risk_values = [float(row.get("risk_score", 0.0) or 0.0) for row in selected]
            info = {
                "count": len(selected),
                "asr_attack_mean": mean(asr_values) if asr_values else 0.0,
                "risk_score_mean": mean(risk_values) if risk_values else 0.0,
                "num_victim_failures_sum": sum(int(row.get("num_victim_failures", 0) or 0) for row in selected),
            }
            phase_rows[threshold_key][attack] = info
            if selected:
                acceptance_rows.append(
                    {
                        "id": f"matrix_{threshold_key}_{attack}",
                        "suite": suite_name,
                        "attack": attack,
                        "dataset_name": dataset_name,
                        "asr_attack": info["asr_attack_mean"],
                        "risk_score": info["risk_score_mean"],
                        "num_victim_failures": info["num_victim_failures_sum"],
                    }
                )
    external_analysis = _load_paper_analysis()
    external_rows = list(external_analysis.get("rows", [])) if external_analysis else []
    acceptance_rows.extend(row for row in external_rows if str(row.get("suite", "")) in {"E0_smoke", "E4_ablation"})
    return phase_rows, acceptance_rows, external_rows


# 组装 `classic phase 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _classic_phase_payload(phase_rows: dict[str, dict[str, dict[str, Any]]], acceptance_result: dict[str, Any], external_rows: list[dict[str, Any]]) -> dict[str, Any]:
    acceptance_findings = list(acceptance_result.get("findings", []))
    phase_ok_map = {
        str(item.get("phase", "")).upper(): bool(item.get("ok", False))
        for item in acceptance_findings
        if "ok" in item
    }
    return {
        "E1": {"ok": bool(phase_ok_map.get("E1", False)), "rows": phase_rows.get("e1", {})},
        "E2": {"ok": bool(phase_ok_map.get("E2", False)), "rows": phase_rows.get("e2", {})},
        "E4": {
            "ok": bool(acceptance_result.get("e4_ok", False)),
            "available": any(str(row.get("suite", "")) == "E4_ablation" for row in external_rows),
            "source": str(PAPER_ANALYSIS_PATH) if external_rows else "",
        },
    }


# 汇总 `classic acceptance 摘要`，从运行记录和指标中提炼页面展示所需的分析结果。
def _classic_acceptance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = _load_thresholds()
    if not thresholds:
        return {
            "available": False,
            "passed": False,
            "findings": ["paper_acceptance_thresholds.json not found"],
            "phases": {},
        }

    phase_rows, acceptance_rows, external_rows = _classic_acceptance_rows(rows)
    acceptance_result = evaluate_acceptance({"rows": acceptance_rows}, thresholds)
    acceptance_findings = list(acceptance_result.get("findings", []))
    phases = _classic_phase_payload(phase_rows, acceptance_result, external_rows)
    findings = [str(item.get("message", "")) for item in acceptance_findings if str(item.get("message", "")).strip()]

    return {
        "available": True,
        "passed": bool(acceptance_result.get("passed", False)),
        "findings": findings,
        "phases": phases,
        "external_analysis_path": str(PAPER_ANALYSIS_PATH) if external_rows else "",
        "acceptance_result": acceptance_result,
    }


# 执行 `aggregate` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                str(row.get("attack", "")),
                str(row.get("dataset_name", "")),
                str(row.get("mode_name", "")),
                str(row.get("model_adapter", "")),
            ]
        )
        grouped[key].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        success_items = [item for item in items if str(item.get("job_status", "")) == "success"]
        metrics = {
            "asr_attack": [float(item.get("asr_attack", 0.0) or 0.0) for item in success_items],
            "asr_defended": [float(item.get("asr_defended", 0.0) or 0.0) for item in success_items],
            "defense_gain": [float(item.get("defense_gain", 0.0) or 0.0) for item in success_items],
            "risk_score": [float(item.get("risk_score", 0.0) or 0.0) for item in success_items],
            "attacked_error_rate@1": [float(item.get("attacked_error_rate@1", 0.0) or 0.0) for item in success_items],
        }
        attack, dataset_name, mode_name, model_adapter = key.split("|")
        aggregate_rows.append(
            {
                "attack": attack,
                "dataset_name": dataset_name,
                "mode_name": mode_name,
                "model_adapter": model_adapter,
                "run_count": len(items),
                "success_count": len(success_items),
                "failed_count": len(items) - len(success_items),
                "asr_attack_mean": mean(metrics["asr_attack"]) if metrics["asr_attack"] else 0.0,
                "asr_attack_std": pstdev(metrics["asr_attack"]) if len(metrics["asr_attack"]) > 1 else 0.0,
                "asr_defended_mean": mean(metrics["asr_defended"]) if metrics["asr_defended"] else 0.0,
                "defense_gain_mean": mean(metrics["defense_gain"]) if metrics["defense_gain"] else 0.0,
                "risk_score_mean": mean(metrics["risk_score"]) if metrics["risk_score"] else 0.0,
                "attacked_error_rate@1_mean": mean(metrics["attacked_error_rate@1"]) if metrics["attacked_error_rate@1"] else 0.0,
                "quality_invalid_count": sum(1 for item in success_items if not bool(item.get("metric_quality_valid", True))),
                "quality_flag_count": sum(int(item.get("num_metric_quality_flags", 0) or 0) for item in success_items),
            }
        )

    classic_by_attack_dataset: list[dict[str, Any]] = []
    for dataset_name in ("coco_subset", "flickr30k"):
        for attack in PAPER_ATTACKS:
            selected = [
                row
                for row in rows
                if row.get("dataset_name") == dataset_name
                and row.get("attack") == attack
                and row.get("mode_name") == "defense"
                and row.get("model_adapter") in CLASSIC_MODELS
                and str(row.get("job_status", "")) == "success"
            ]
            classic_by_attack_dataset.append(
                {
                    "dataset_name": dataset_name,
                    "attack": attack,
                    "count": len(selected),
                    "asr_attack_mean": mean([float(row.get("asr_attack", 0.0) or 0.0) for row in selected]) if selected else 0.0,
                    "risk_score_mean": mean([float(row.get("risk_score", 0.0) or 0.0) for row in selected]) if selected else 0.0,
                    "num_victim_failures_sum": sum(int(row.get("num_victim_failures", 0) or 0) for row in selected),
                }
            )

    return {
        "generated_at": _now_iso(),
        "row_count": len(rows),
        "success_count": sum(1 for row in rows if str(row.get("job_status", "")) == "success"),
        "failed_count": sum(1 for row in rows if str(row.get("job_status", "")) != "success"),
        "aggregate_rows": aggregate_rows,
        "classic_paper_summary": classic_by_attack_dataset,
        "classic_acceptance": _classic_acceptance_summary(rows),
    }


# 整理 `行记录 来源 spec` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _row_from_spec(job_spec: dict[str, Any], *, seed_base: int, row_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(seed_base) + int(job_spec["repeat_index"]) - 1
    experiment_id, payload = _job_payload(
        attack=str(job_spec["attack"]),
        config_path=str(job_spec["config_path"]),
        dataset_name=str(job_spec["dataset_name"]),
        mode_name=str(job_spec["mode_name"]),
        model_adapter=str(job_spec["model_adapter"]),
        repeat_index=int(job_spec["repeat_index"]),
        seed=seed,
    )
    row: dict[str, Any] = {
        "submitted_at": _now_iso(),
        "attack": str(job_spec["attack"]),
        "dataset_name": str(job_spec["dataset_name"]),
        "mode_name": str(job_spec["mode_name"]),
        "model_adapter": str(job_spec["model_adapter"]),
        "repeat_index": int(job_spec["repeat_index"]),
        "seed": seed,
        "experiment_id": experiment_id,
        "job_index": row_index,
    }
    return row, payload


# 执行 `active 任务 by 实验` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _active_jobs_by_experiment(api: ApiClient) -> dict[str, dict[str, Any]]:
    try:
        data = api.get("/jobs?page=1&page_size=200")
    except (OSError, RuntimeError, TypeError, ValueError, requests.RequestException):
        return {}
    active: dict[str, dict[str, Any]] = {}
    for job in data.get("items", []):
        if str(job.get("status", "")) not in {"queued", "running"}:
            continue
        try:
            override = json.loads(str(job.get("override_json", "") or "{}"))
        except json.JSONDecodeError:
            override = {}
        runner = override.get("runner", {}) if isinstance(override.get("runner", {}), dict) else {}
        dataset = override.get("dataset", {}) if isinstance(override.get("dataset", {}), dict) else {}
        experiment_id = str(runner.get("experiment_id", "") or dataset.get("benchmark_tag", "")).strip()
        if experiment_id:
            active[experiment_id] = job
    return active


# 整理 `finalize 任务 行记录` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _finalize_job_row(api: ApiClient, row: dict[str, Any], final_job: dict[str, Any]) -> None:
    row.pop("_deadline_ts", None)
    row.pop("_poll_error_count", None)
    row["job_status"] = str(final_job.get("status", ""))
    row["finished_at"] = _now_iso()
    if row["job_status"] == "success":
        run_id = str(final_job.get("run_id", ""))
        row["run_id"] = run_id
        summary = api.get(f"/runs/{run_id}/summary")
        row["asr_attack"] = float(summary.get("asr_attack", summary.get("asr", 0.0)) or 0.0)
        row["conditional_asr_attack"] = float(summary.get("conditional_asr_attack", row["asr_attack"]) or 0.0)
        row["attacked_error_rate@1"] = float(summary.get("attacked_error_rate@1", summary.get("unconditional_asr_attack", 0.0)) or 0.0)
        row["asr_definition"] = str(summary.get("asr_definition", "conditional_clean_top1_drop"))
        row["asr_defended"] = float(summary.get("asr_defended", 0.0) or 0.0)
        row["defense_gain"] = float(summary.get("defense_gain", 0.0) or 0.0)
        row["risk_score"] = float(summary.get("risk_score", 0.0) or 0.0)
        metric_quality = summary.get("metric_quality", {}) if isinstance(summary.get("metric_quality", {}), dict) else {}
        row["metric_quality_valid"] = bool(metric_quality.get("valid_for_attack_strength_claim", True))
        row["metric_quality_flags"] = metric_quality.get("flags", [])
        row["num_metric_quality_flags"] = len(row["metric_quality_flags"]) if isinstance(row["metric_quality_flags"], list) else 0
        row["num_victim_failures"] = int(summary.get("num_victim_failures", 0) or 0)
        row["victim_model_adapters"] = summary.get("victim_model_adapters", [])
        row["benchmark_tag"] = str(summary.get("benchmark_tag", ""))
    else:
        logs = api.get(f"/jobs/{row['job_id']}/logs?page=1&page_size=200")
        row["logs_tail"] = logs.get("items", [])[-20:]


# 整理 `finalize exception 行记录` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _finalize_exception_row(api: ApiClient, row: dict[str, Any], exc: Exception) -> None:
    row.pop("_deadline_ts", None)
    row.pop("_poll_error_count", None)
    try:
        api.post(f"/jobs/{row['job_id']}/cancel", {})
        row["cancel_requested"] = True
    except (OSError, RuntimeError, TypeError, ValueError, requests.RequestException):
        row["cancel_requested"] = False
    row["job_status"] = "failed"
    row["finished_at"] = _now_iso()
    row["exception"] = str(exc)
    row["traceback"] = traceback.format_exc()
    try:
        logs = api.get(f"/jobs/{row['job_id']}/logs?page=1&page_size=200")
        row["logs_tail"] = logs.get("items", [])[-20:]
    except (OSError, RuntimeError, TypeError, ValueError, requests.RequestException) as log_exc:
        row["logs_error"] = str(log_exc)


# 写出 `矩阵 outputs`，保证后续报告、页面或复现实验能读取。
def _write_matrix_outputs(
    *,
    rows: list[dict[str, Any]],
    status: dict[str, Any],
    rows_path: Path,
    summary_path: Path,
    status_path: Path,
) -> None:
    rows_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(_aggregate(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


# 解析 `args`，把文本或载荷转换成可校验的字段。
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", default="coco_subset,flickr30k,mini_flickr")
    parser.add_argument("--modes", default="standard,defense")
    parser.add_argument("--attacks", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-items", type=int, default=0, help="Override per-dataset max_items. Use 5000 for the COCO 5k evaluation split.")
    parser.add_argument("--max-pairs", type=int, default=-1, help="Override runner.max_pairs. Use 0 to disable downsampling; -1 keeps script defaults.")
    parser.add_argument("--seed-base", type=int, default=20260417)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--concurrency", type=int, default=1, help="Number of API jobs to keep in flight.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


# 构建 `矩阵 paths`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _matrix_paths(out_dir_arg: str) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir_arg).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "status.json", out_dir / "rows.json", out_dir / "summary.json"


# 构建 `矩阵 filters`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _matrix_filters(args: argparse.Namespace) -> tuple[list[str], list[str], list[Any], list[str]]:
    attack_filter = {item.strip() for item in str(args.attacks or "").split(",") if item.strip()}
    model_filter = {item.strip() for item in str(args.models or "").split(",") if item.strip()}
    dataset_names = [item.strip() for item in str(args.datasets or "").split(",") if item.strip()]
    mode_names = [item.strip() for item in str(args.modes or "").split(",") if item.strip()]
    attack_specs = [item for item in ATTACK_SPECS if not attack_filter or item.attack in attack_filter]
    model_adapters = [item for item in MAIN_MODELS if not model_filter or item in model_filter]
    return dataset_names, mode_names, attack_specs, model_adapters


# 构建 `矩阵 计划`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _matrix_plan(
    *,
    dataset_names: list[str],
    mode_names: list[str],
    attack_specs: list[Any],
    model_adapters: list[str],
    repeats: int,
    done_keys: set[tuple[str, str, str, str, int]],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        for mode_name in mode_names:
            for attack_spec in attack_specs:
                for model_adapter in model_adapters:
                    for repeat_index in range(1, max(1, int(repeats)) + 1):
                        key = (attack_spec.attack, dataset_name, mode_name, model_adapter, repeat_index)
                        if key in done_keys:
                            continue
                        plan.append(
                            {
                                "attack": attack_spec.attack,
                                "config_path": attack_spec.config_path,
                                "dataset_name": dataset_name,
                                "mode_name": mode_name,
                                "model_adapter": model_adapter,
                                "repeat_index": repeat_index,
                            }
                        )
    return plan


# 构建 `initial 矩阵 状态`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _initial_matrix_status(
    *,
    args: argparse.Namespace,
    overview: dict[str, Any],
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    dataset_names: list[str],
    mode_names: list[str],
    attack_specs: list[Any],
    model_adapters: list[str],
) -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "started_at": _now_iso(),
        "status": "running",
        "api_base": args.api_base,
        "overview_generated_at": overview.get("generated_at"),
        "expected_total": len(rows) + len(plan),
        "completed": len(rows),
        "remaining": len(plan),
        "datasets": dataset_names,
        "modes": mode_names,
        "attacks": [item.attack for item in attack_specs],
        "models": model_adapters,
        "repeats": int(args.repeats),
        "max_items_override": DATASET_MAX_ITEMS_OVERRIDE,
        "max_pairs_override": MAX_PAIRS_OVERRIDE,
        "concurrency": max(1, int(args.concurrency or 1)),
    }


# 整理 `adopt active rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _adopt_active_rows(
    *,
    api: ApiClient,
    plan: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    concurrency: int,
    timeout_seconds: int,
    seed_base: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    active_rows: list[dict[str, Any]] = []
    api_active_jobs = _active_jobs_by_experiment(api)
    adopted_plan_indices: set[int] = set()
    for plan_index, job_spec in enumerate(plan):
        row, _payload = _row_from_spec(job_spec, seed_base=seed_base, row_index=len(rows) + len(active_rows) + 1)
        active_job = api_active_jobs.get(str(row["experiment_id"]))
        if not active_job:
            continue
        row["job_id"] = str(active_job.get("id", ""))
        row["submitted_at"] = str(active_job.get("created_at", row["submitted_at"]) or row["submitted_at"])
        row["adopted_from_api"] = True
        row["_deadline_ts"] = time.time() + timeout_seconds
        active_rows.append(row)
        adopted_plan_indices.add(plan_index)
        if len(active_rows) >= concurrency:
            break
    if not adopted_plan_indices:
        return plan, active_rows, 0
    remaining_plan = [job_spec for plan_index, job_spec in enumerate(plan) if plan_index not in adopted_plan_indices]
    return remaining_plan, active_rows, len(adopted_plan_indices)


# 执行 `submit 任务 until full` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _submit_jobs_until_full(
    *,
    api: ApiClient,
    plan: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    next_plan_index: int,
    concurrency: int,
    timeout_seconds: int,
    seed_base: int,
) -> int:
    while next_plan_index < len(plan) and len(active_rows) < concurrency:
        row, payload = _row_from_spec(plan[next_plan_index], seed_base=seed_base, row_index=len(rows) + len(active_rows) + 1)
        created = api.create_job(payload)
        row["job_id"] = str(created.get("id", ""))
        row["_deadline_ts"] = time.time() + timeout_seconds
        active_rows.append(row)
        next_plan_index += 1
    return next_plan_index


# 整理 `last 行记录 状态` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _last_row_status(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "attack": row["attack"],
        "dataset_name": row["dataset_name"],
        "mode_name": row["mode_name"],
        "model_adapter": row["model_adapter"],
        "repeat_index": row["repeat_index"],
        "job_status": row.get("job_status", ""),
        "run_id": row.get("run_id", ""),
    }


# 整理 `poll active rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _poll_active_rows(api: ApiClient, rows: list[dict[str, Any]], active_rows: list[dict[str, Any]], status: dict[str, Any]) -> bool:
    progressed = False
    for row in list(active_rows):
        try:
            if time.time() > float(row.get("_deadline_ts", 0.0) or 0.0):
                raise TimeoutError(f"job timed out: {row.get('job_id', '')}")
            final_job = api.get_job(str(row["job_id"]))
            if str(final_job.get("status", "")) not in TERMINAL_JOB_STATUSES:
                continue
            _finalize_job_row(api, row, final_job)
        except (KeyError, OSError, RuntimeError, TimeoutError, TypeError, ValueError, requests.RequestException) as exc:
            if not isinstance(exc, TimeoutError):
                row["_poll_error_count"] = int(row.get("_poll_error_count", 0) or 0) + 1
                row["last_poll_error"] = str(exc)
                if int(row["_poll_error_count"]) <= 8:
                    continue
            _finalize_exception_row(api, row, exc)

        active_rows.remove(row)
        rows.append(row)
        progressed = True
        status["last_row"] = _last_row_status(row)
    return progressed


# 构建 `update 矩阵 状态`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _update_matrix_status(
    *,
    status: dict[str, Any],
    rows: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    next_plan_index: int,
) -> None:
    status["completed"] = len(rows)
    status["submitted"] = len(rows) + len(active_rows)
    status["remaining"] = max(0, len(plan) - next_plan_index)
    status["in_flight"] = [
        {
            "job_id": row.get("job_id", ""),
            "attack": row.get("attack", ""),
            "dataset_name": row.get("dataset_name", ""),
            "mode_name": row.get("mode_name", ""),
            "model_adapter": row.get("model_adapter", ""),
            "repeat_index": row.get("repeat_index", ""),
            "adopted_from_api": bool(row.get("adopted_from_api", False)),
        }
        for row in active_rows
    ]


# 构建 `运行记录 矩阵 loop`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _run_matrix_loop(
    *,
    api: ApiClient,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    status: dict[str, Any],
    rows_path: Path,
    summary_path: Path,
    status_path: Path,
) -> None:
    concurrency = max(1, int(args.concurrency or 1))
    timeout_seconds = max(30, int(args.timeout_seconds))
    poll_seconds = max(1.0, float(args.poll_seconds))
    plan, active_rows, adopted_count = _adopt_active_rows(
        api=api,
        plan=plan,
        rows=rows,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        seed_base=int(args.seed_base),
    )
    if adopted_count:
        status["adopted_active_jobs"] = adopted_count
    next_plan_index = 0
    while next_plan_index < len(plan) or active_rows:
        next_plan_index = _submit_jobs_until_full(
            api=api,
            plan=plan,
            rows=rows,
            active_rows=active_rows,
            next_plan_index=next_plan_index,
            concurrency=concurrency,
            timeout_seconds=timeout_seconds,
            seed_base=int(args.seed_base),
        )
        progressed = _poll_active_rows(api, rows, active_rows, status)
        _update_matrix_status(
            status=status,
            rows=rows,
            active_rows=active_rows,
            plan=plan,
            next_plan_index=next_plan_index,
        )
        if progressed:
            _write_matrix_outputs(rows=rows, status=status, rows_path=rows_path, summary_path=summary_path, status_path=status_path)
        else:
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        if next_plan_index < len(plan) or active_rows:
            time.sleep(poll_seconds)


# 写出 `failed 矩阵 状态`，保证后续报告、页面或复现实验能读取。
def _write_failed_matrix_status(status: dict[str, Any], status_path: Path, exc: Exception) -> None:
    status["status"] = "failed"
    status["ended_at"] = _now_iso()
    status["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


# 作为 `run_server_exhaustive_matrix.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    args = _parse_args()

    global DATASET_MAX_ITEMS_OVERRIDE, MAX_PAIRS_OVERRIDE
    DATASET_MAX_ITEMS_OVERRIDE = max(0, int(args.max_items or 0))
    MAX_PAIRS_OVERRIDE = int(args.max_pairs)

    status_path, rows_path, summary_path = _matrix_paths(args.out_dir)
    dataset_names, mode_names, attack_specs, model_adapters = _matrix_filters(args)

    api = ApiClient(args.api_base, timeout=120.0)
    overview = api.get("/system/overview")

    rows = _load_existing_rows(rows_path) if args.resume else []
    done_keys = {_row_key(row) for row in rows}
    plan = _matrix_plan(
        dataset_names=dataset_names,
        mode_names=mode_names,
        attack_specs=attack_specs,
        model_adapters=model_adapters,
        repeats=int(args.repeats),
        done_keys=done_keys,
    )
    status = _initial_matrix_status(
        args=args,
        overview=overview,
        rows=rows,
        plan=plan,
        dataset_names=dataset_names,
        mode_names=mode_names,
        attack_specs=attack_specs,
        model_adapters=model_adapters,
    )
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        _run_matrix_loop(
            api=api,
            args=args,
            rows=rows,
            plan=plan,
            status=status,
            rows_path=rows_path,
            summary_path=summary_path,
            status_path=status_path,
        )
        status["status"] = "completed"
        status["ended_at"] = _now_iso()
        _write_matrix_outputs(rows=rows, status=status, rows_path=rows_path, summary_path=summary_path, status_path=status_path)
        return 0
    except (KeyError, OSError, RuntimeError, TimeoutError, TypeError, ValueError, requests.RequestException) as exc:
        _write_failed_matrix_status(status, status_path, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
