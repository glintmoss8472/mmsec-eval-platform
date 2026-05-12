from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Tuple


StageDef = Tuple[str, str, int]


JOB_STAGE_TEMPLATES: dict[str, list[StageDef]] = {
    "run_vlr": [
        ("queued", "排队中", 5),
        ("model_preflight", "模型预检查", 16),
        ("config_validation", "配置校验", 26),
        ("dataset_loading", "数据集装载", 38),
        ("attack_execution", "执行攻击", 58),
        ("victim_evaluation", "受测模型评测", 78),
        ("result_aggregation", "结果汇总", 90),
        ("report_writing", "报告写入", 97),
        ("completed", "完成", 100),
    ],
    "run_eval": [
        ("queued", "排队中", 5),
        ("model_preflight", "模型预检查", 16),
        ("config_validation", "配置校验", 26),
        ("dataset_loading", "数据集装载", 38),
        ("attack_execution", "执行攻击", 60),
        ("victim_evaluation", "受测模型评测", 80),
        ("result_aggregation", "结果汇总", 92),
        ("report_writing", "报告写入", 97),
        ("completed", "完成", 100),
    ],
    "run_vqa": [
        ("queued", "排队中", 5),
        ("model_preflight", "模型预检查", 16),
        ("config_validation", "配置校验", 26),
        ("dataset_loading", "VQA 样本装载", 38),
        ("attack_execution", "执行攻击与问答生成", 82),
        ("result_aggregation", "结果汇总", 92),
        ("report_writing", "报告写入", 97),
        ("completed", "完成", 100),
    ],
    "run_caption": [
        ("queued", "排队中", 5),
        ("model_preflight", "模型预检查", 16),
        ("config_validation", "配置校验", 26),
        ("dataset_loading", "图像描述样本装载", 38),
        ("attack_execution", "执行攻击与描述生成", 82),
        ("result_aggregation", "结果汇总", 92),
        ("report_writing", "报告写入", 97),
        ("completed", "完成", 100),
    ],
    "train_advclip": [
        ("queued", "排队中", 5),
        ("model_preflight", "模型预检查", 18),
        ("config_validation", "配置校验", 28),
        ("dataset_loading", "数据集装载", 38),
        ("attack_execution", "执行补丁训练", 78),
        ("result_aggregation", "结果汇总", 90),
        ("report_writing", "报告写入", 97),
        ("completed", "完成", 100),
    ],
    "generate_sample_assets": [
        ("queued", "排队中", 5),
        ("model_preflight", "代理模型检查", 16),
        ("config_validation", "配置校验", 26),
        ("dataset_loading", "来源数据集读取", 42),
        ("attack_execution", "生成对抗样本", 82),
        ("result_aggregation", "样本资产入库", 94),
        ("completed", "完成", 100),
    ],
}


DEFAULT_TEMPLATE: list[StageDef] = [
    ("queued", "排队中", 5),
    ("config_validation", "配置校验", 30),
    ("attack_execution", "执行任务", 75),
    ("result_aggregation", "结果汇总", 92),
    ("completed", "完成", 100),
]


DEFAULT_DURATION_HINTS: dict[str, int] = {
    "run_vlr": 95,
    "run_eval": 90,
    "run_vqa": 100,
    "run_caption": 110,
    "train_advclip": 320,
    "generate_sample_assets": 90,
    "dataset_prepare": 80,
}


PAIR_PROGRESS_RE = re.compile(r"已完成\s*(?P<done>\d+)\s*/\s*(?P<total>\d+)\s*对图文配对")


def stage_template(job_type: str) -> list[StageDef]:
    return JOB_STAGE_TEMPLATES.get(str(job_type or ""), DEFAULT_TEMPLATE)


def initial_stage_rows(job_type: str, now_iso: str) -> list[dict[str, Any]]:
    rows = []
    for order, (stage_key, stage_label, progress_percent) in enumerate(stage_template(job_type), start=1):
        rows.append(
            {
                "stage_key": stage_key,
                "stage_label": stage_label,
                "stage_order": order,
                "state": "running" if stage_key == "queued" else "pending",
                "progress_percent": float(progress_percent if stage_key == "queued" else 0),
                "message": "任务已提交，等待 worker 执行。" if stage_key == "queued" else "",
                "updated_at": now_iso,
            }
        )
    return rows


def parse_iso_ts(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def duration_seconds(started_at: str | None, finished_at: str | None) -> float:
    start_dt = parse_iso_ts(started_at)
    end_dt = parse_iso_ts(finished_at)
    if not start_dt or not end_dt:
        return 0.0
    return max(0.0, (end_dt - start_dt).total_seconds())


def median_duration_seconds(job_type: str, durations: list[float]) -> float:
    values = [float(item) for item in durations if float(item) > 0]
    if values:
        return float(median(values))
    return float(DEFAULT_DURATION_HINTS.get(job_type, 90))


def parse_pair_progress(message: str | None) -> tuple[int, int] | None:
    text = str(message or "").strip()
    if not text:
        return None
    match = PAIR_PROGRESS_RE.search(text)
    if not match:
        return None
    done = int(match.group("done"))
    total = int(match.group("total"))
    if done < 0 or total <= 0 or done > total:
        return None
    return done, total



def pair_progress_percent(message: str | None) -> float:
    parsed = parse_pair_progress(message)
    if parsed is None:
        return 0.0
    done, total = parsed
    if total <= 0:
        return 0.0
    return float(round((float(done) / float(total)) * 100.0, 2))

def estimate_pair_eta_seconds(*, elapsed_seconds: float, stage_message: str | None) -> float | None:
    parsed = parse_pair_progress(stage_message)
    if parsed is None:
        return None
    done, total = parsed
    if done <= 0 or total <= done or float(elapsed_seconds) <= 0:
        return None
    rate = float(done) / float(elapsed_seconds)
    if rate <= 0:
        return None
    remaining_pairs = float(total - done)
    return float(round(max(0.0, remaining_pairs / rate), 2))


def estimate_eta_seconds(
    *,
    job_type: str,
    status: str,
    queue_position: int,
    elapsed_seconds: float,
    progress_percent: float,
    recent_durations: list[float],
    worker_count: int,
    stage_message: str = "",
) -> float:
    base = median_duration_seconds(job_type, recent_durations)
    workers = max(1, int(worker_count))
    if status in {"success", "failed", "cancelled"}:
        return 0.0
    if status == "queued":
        rounds = max(1.0, float(queue_position) / float(workers))
        return float(round(base * rounds, 2))
    pair_eta = estimate_pair_eta_seconds(elapsed_seconds=elapsed_seconds, stage_message=stage_message)
    if pair_eta is not None:
        return pair_eta
    normalized_progress = max(1.0, float(progress_percent))
    inferred_total = max(base, float(elapsed_seconds) / max(0.05, normalized_progress / 100.0))
    return float(round(max(0.0, inferred_total - float(elapsed_seconds)), 2))


def estimated_ready_at(eta_seconds: float) -> str:
    if float(eta_seconds) <= 0:
        return ""
    ready_at = datetime.now(timezone.utc) + timedelta(seconds=float(eta_seconds))
    return ready_at.isoformat().replace("+00:00", "Z")
