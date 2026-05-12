# 文件说明：该文件属于自动化测试，集中实现 test sample store 相关逻辑。
from __future__ import annotations

from pathlib import Path

import numpy as np

from mmsec_eval.sample_store.manager import SampleStoreManager
from mmsec_eval.types import AttackedSample, EvalRecord, JudgeResult, ModelOutput, Sample


# 验证 `样本 store manager` 场景，防止相关行为在后续修改中退化。
def test_sample_store_manager(tmp_path: Path):
    run_dir = tmp_path / "artifacts" / "runs" / "r1"
    mgr = SampleStoreManager(str(run_dir), save_images=True, save_traces=True)

    clean = Sample("s1", np.zeros((16, 16, 3), dtype=np.float32), "a circle", target_text="square")
    adv_s = Sample("s1", np.ones((16, 16, 3), dtype=np.float32) * 0.1, "a square", target_text="square")
    attacked = AttackedSample(sample=adv_s, perturbation_l2=1.0, perturbation_linf=0.1)

    rec = EvalRecord(
        sample=clean,
        attacked=attacked,
        pred_clean=ModelOutput("detected object: circle"),
        pred_adv=ModelOutput("detected object: square"),
        judge=JudgeResult(True, "target_injected"),
        metrics={"perturbation_l2": 1.0},
    )

    refs = mgr.persist_record(rec)
    idx = mgr.flush()

    assert Path(idx).exists()
    assert Path(refs["clean_image"]).exists()
    assert Path(refs["adv_image"]).exists()
    assert Path(refs["case_bundle"]).exists()
