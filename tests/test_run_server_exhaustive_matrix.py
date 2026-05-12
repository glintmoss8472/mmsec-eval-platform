# 文件说明：该文件属于自动化测试，集中实现 test run server exhaustive matrix 相关逻辑。
from __future__ import annotations

import importlib.util
from pathlib import Path


# 中文注释：封装 _load_module 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _load_module():
    path = Path("scripts/run_server_exhaustive_matrix.py").resolve()
    spec = importlib.util.spec_from_file_location("run_server_exhaustive_matrix", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 中文注释：验证 test_classic_acceptance_uses_matrix_e1_e2_and_external_e4 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_classic_acceptance_uses_matrix_e1_e2_and_external_e4():
    module = _load_module()
    thresholds = module._load_thresholds()

    rows = []
    for dataset_name, threshold_key in (("coco_subset", "e1"), ("flickr30k", "e2")):
        minimums = dict(thresholds[threshold_key]["minimum_asr_attack"])
        synthetic_asr = {
            "advclip": float(minimums["advclip"]) + 0.01,
            "advedm": float(minimums["advedm"]) + 0.02,
            "advedm_plus": float(minimums["advedm_plus"]) + 0.08,
            "tmm": float(minimums["tmm"]) + 0.16,
        }
        for attack, asr_value in synthetic_asr.items():
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "attack": attack,
                    "mode_name": "defense",
                    "model_adapter": "clip_hf",
                    "job_status": "success",
                    "asr_attack": float(asr_value),
                    "defense_gain": 0.12 if attack == "tmm" else 0.16 if attack == "advedm_plus" else 0.0,
                    "risk_score": 0.5,
                    "num_victim_failures": 0,
                }
            )

    result = module._classic_acceptance_summary(rows)

    assert result["available"] is True
    assert result["passed"] is True
    assert result["phases"]["E1"]["ok"] is True
    assert result["phases"]["E2"]["ok"] is True
    assert result["phases"]["E4"]["ok"] is True
    assert result["external_analysis_path"]


# 中文注释：验证 test_dataset_override_supports_flickr1k 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_dataset_override_supports_flickr1k():
    module = _load_module()
    payload = module._dataset_override("flickr1k")

    assert payload["kind"] == "flickr1k"
    assert payload["captions_file"] == "captions_index_single.jsonl"
