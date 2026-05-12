# 文件说明：该文件属于自动化测试，集中实现 test sample store defended bundle 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmsec_eval.sample_store.manager import SampleStoreManager
from mmsec_eval.types import AttackedSample, EvalRecord, JudgeResult, ModelOutput, Sample


# 中文注释：验证 test_sample_store_persists_defended_outputs 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_sample_store_persists_defended_outputs(tmp_path: Path):
    run_dir = tmp_path / "run"
    mgr = SampleStoreManager(run_dir=str(run_dir), save_images=True, save_traces=True, dataset_tag="toy", model_tag="clip_hf")

    clean = Sample(sample_id="s1", image=np.zeros((32, 32, 3), dtype=np.float32), text="clean", target_text="", metadata={})
    adv_s = Sample(sample_id="s1", image=np.ones((32, 32, 3), dtype=np.float32) * 0.1, text="adv", target_text="", metadata={})
    def_s = Sample(sample_id="s1", image=np.ones((32, 32, 3), dtype=np.float32) * 0.05, text="def", target_text="", metadata={})
    attacked = AttackedSample(sample=adv_s, perturbation_l0=10, perturbation_l2=1.0, perturbation_linf=0.1)

    rec = EvalRecord(
        sample=clean,
        attacked=attacked,
        pred_clean=ModelOutput(text="clean", score=0.1),
        pred_adv=ModelOutput(text="adv", score=0.2),
        judge=JudgeResult(success=True, reason="ok"),
        metrics={"perturbation_l2": 1.0},
        diagnostics={},
    )

    refs = mgr.persist_record(
        rec,
        defended_sample=def_s,
        pred_defended=ModelOutput(text="def", score=0.15),
        defense_refs={"defense_trace": str(run_dir / "trace.json")},
        defense_gain_sample=0.5,
    )
    mgr.flush()

    bundle = json.loads((run_dir / "cases" / "s1" / "case_bundle.json").read_text(encoding="utf-8"))
    assert "defended" in bundle.get("outputs", {})
    assert "defended_image" in bundle.get("artifact_refs", {})
    assert "defense_trace" in bundle.get("artifact_refs", {})

    rows = (run_dir / "cases_index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert rows
    idx = json.loads(rows[0])
    assert float(idx.get("defense_gain_sample", 0.0)) == 0.5
    assert refs.get("defended_image", "")
