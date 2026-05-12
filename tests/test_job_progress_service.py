from __future__ import annotations

from mmsec_api.services.job_progress import estimate_eta_seconds, estimate_pair_eta_seconds, parse_pair_progress


def test_parse_pair_progress_extracts_counts():
    parsed = parse_pair_progress("正在评测正常输入在各受测模型上的表现：openai_qwen3_vl，已完成 192/12100 对图文配对。")
    assert parsed == (192, 12100)


def test_estimate_pair_eta_seconds_uses_real_pair_rate():
    eta = estimate_pair_eta_seconds(
        elapsed_seconds=1311.0,
        stage_message="正在评测正常输入在各受测模型上的表现：openai_qwen3_vl，已完成 192/12100 对图文配对。",
    )
    assert eta is not None
    assert 81000 <= eta <= 81500


def test_estimate_eta_seconds_prefers_pair_progress_over_stage_percent():
    eta = estimate_eta_seconds(
        job_type="run_vlr",
        status="running",
        queue_position=0,
        elapsed_seconds=1311.0,
        progress_percent=46.16,
        recent_durations=[2800.0, 3100.0],
        worker_count=1,
        stage_message="正在评测正常输入在各受测模型上的表现：openai_qwen3_vl，已完成 192/12100 对图文配对。",
    )
    assert eta > 80000


def test_estimate_eta_seconds_falls_back_when_pair_progress_is_missing():
    eta = estimate_eta_seconds(
        job_type="run_vlr",
        status="running",
        queue_position=0,
        elapsed_seconds=100.0,
        progress_percent=50.0,
        recent_durations=[200.0, 180.0],
        worker_count=1,
        stage_message="正在汇总多模型指标、风险分数和样本级摘要。",
    )
    assert eta == 100.0
