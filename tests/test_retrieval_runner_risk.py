from __future__ import annotations

import pytest

from mmsec_eval.runner.retrieval_runner import (
    _summarize_attack_outcomes,
    _victim_conditional_asr_at_1,
    _victim_stage_asr_at_1,
    _victim_transfer_score,
)


def test_victim_stage_asr_at_1_averages_ir_and_tr():
    metrics = {"ir_asr@1": 0.8, "tr_asr@1": 0.4}
    assert _victim_stage_asr_at_1(metrics) == pytest.approx(0.6)


def test_victim_conditional_asr_at_1_averages_ir_and_tr():
    metrics = {"ir_cond_asr@1": 0.25, "tr_cond_asr@1": 0.75}
    assert _victim_conditional_asr_at_1(metrics) == pytest.approx(0.5)


def test_victim_transfer_score_uses_cross_victim_coverage():
    victim_metrics = {
        "clip_hf": {"attacked": {"ir_asr@1": 0.8, "tr_asr@1": 0.6}},
        "blip_itm": {"attacked": {"ir_asr@1": 0.3, "tr_asr@1": 0.1}},
        "vilt_itm": {"attacked": {"ir_asr@1": 0.05, "tr_asr@1": 0.05}},
    }

    score, victim_asrs = _victim_transfer_score(
        victim_metrics,
        ["clip_hf", "blip_itm", "vilt_itm"],
        threshold=0.2,
    )

    assert victim_asrs == pytest.approx([0.7, 0.2, 0.05])
    assert score == pytest.approx(2 / 3)


def test_victim_transfer_score_prefers_conditional_asr_when_available():
    victim_metrics = {
        "qwen25_vl": {
            "attacked": {"ir_asr@1": 0.9545, "tr_asr@1": 0.9545},
            "conditional": {"ir_cond_asr@1": 0.0, "tr_cond_asr@1": 0.0},
        },
        "gemma3_12b": {
            "attacked": {"ir_asr@1": 0.9545, "tr_asr@1": 0.9545},
            "conditional": {"ir_cond_asr@1": 0.0, "tr_cond_asr@1": 0.0},
        },
    }

    score, victim_asrs = _victim_transfer_score(
        victim_metrics,
        ["qwen25_vl", "gemma3_12b"],
        threshold=0.2,
    )

    assert victim_asrs == pytest.approx([0.0, 0.0])
    assert score == pytest.approx(0.0)


def test_victim_transfer_score_does_not_reuse_attack_mean_for_single_victim():
    victim_metrics = {
        "clip_hf": {"attacked": {"ir_asr@1": 0.9, "tr_asr@1": 0.7}},
    }

    score, victim_asrs = _victim_transfer_score(
        victim_metrics,
        ["clip_hf"],
        threshold=0.2,
    )

    assert victim_asrs == pytest.approx([0.8])
    assert score == pytest.approx(0.0)


def test_summarize_attack_outcomes_keeps_conditional_asr_separate_from_error_rate():
    victim_metrics = {
        "qwen25_vl": {
            "clean": {"mean_rank_ir": 1.0, "mean_rank_tr": 1.0},
            "attacked": {"ir_asr@1": 0.9545, "tr_asr@1": 0.9545, "mean_rank_ir": 5.0, "mean_rank_tr": 7.0},
            "conditional": {"ir_cond_asr@1": 0.0, "tr_cond_asr@1": 0.0},
        },
        "clip_hf": {
            "clean": {"mean_rank_ir": 1.0, "mean_rank_tr": 2.0},
            "attacked": {"ir_asr@1": 0.5, "tr_asr@1": 0.25, "mean_rank_ir": 3.0, "mean_rank_tr": 4.0},
            "conditional": {"ir_cond_asr@1": 0.2, "tr_cond_asr@1": 0.4},
            "defended_attack": {"ir_asr@1": 0.1, "tr_asr@1": 0.3},
        },
    }

    summary = _summarize_attack_outcomes(
        victim_metrics=victim_metrics,
        victim_names=["qwen25_vl", "clip_hf"],
        eval_scope="joint",
        attack_debug={"l2_values": [1.0, 3.0], "linf_values": [0.1, 0.3], "text_changed_ratio": 0.25},
        transfer_threshold=0.2,
    )

    assert summary["attacked_error_rate_at1"] == pytest.approx((0.9545 + 0.9545 + 0.5 + 0.25) / 4)
    assert summary["conditional_asr_attack"] == pytest.approx(0.15)
    assert summary["asr_attack"] == pytest.approx(0.15)
    assert summary["asr_defended"] == pytest.approx(0.1)
    assert summary["avg_l2_all"] == pytest.approx(2.0)
    assert summary["avg_linf_all"] == pytest.approx(0.2)
    assert summary["text_changed_ratio"] == pytest.approx(0.25)
    assert summary["rank_deltas"] == pytest.approx([5.0, 2.0])
