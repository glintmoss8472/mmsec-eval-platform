# 文件说明：该文件属于自动化测试，集中实现 test run compare api 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

from api_test_utils import make_client


# 写出 `运行记录`，保证后续报告、页面或复现实验能读取。
def _write_run(art: Path, run_id: str, asr: float, asr_def: float):
    run_dir = art / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "task_kind": "vlr",
        "dataset_name": "toy_shapes",
        "attack": "advclip",
        "attack_mode": "A",
        "defense": "sanitize_v1",
        "defense_enabled": True,
        "asr": asr,
        "asr_attack": asr,
        "asr_defended": asr_def,
        "defense_gain": asr - asr_def,
        "risk_score": asr,
        "risk_level": "high",
        "risk_scenario": "retrieval",
        "victims": {
            "clip_hf": {
                "clean": {"ir_r@1": 0.8},
                "attacked": {"ir_r@1": 0.3},
                "defended_attack": {"ir_r@1": 0.5},
                "defended_clean": {"ir_r@1": 0.75},
            }
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report_data.json").write_text(json.dumps({"stage_metrics": {}}, ensure_ascii=False, indent=2), encoding="utf-8")


# 验证 `运行记录 对比 API` 场景，防止相关行为在后续修改中退化。
def test_runs_compare_api(tmp_path: Path, monkeypatch):
    art = tmp_path / "artifacts"
    _write_run(art, "r1", 0.6, 0.4)
    _write_run(art, "r2", 0.7, 0.5)

    with make_client(tmp_path, monkeypatch) as client:
        resp = client.get("/api/v1/runs/compare", params={"run_ids": "r1,r2"})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("run_ids") == ["r1", "r2"]
        cmp_v = (data.get("compare", {}) or {}).get("victims", {})
        assert "clip_hf" in cmp_v
