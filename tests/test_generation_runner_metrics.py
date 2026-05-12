# 文件说明：该文件属于自动化测试，集中实现 test generation runner metrics 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.runner.generation_runner import _caption_metrics, _case_bundle
from mmsec_eval.types import ModelOutput, Sample


# 中文注释：定义 NoProbeModel 的结构化职责，作为自动化测试中状态、配置或行为的边界。
class NoProbeModel:
    # 中文注释：实现 NoProbeModel.object_probe 的核心行为，维护自动化测试在该对象上的调用契约。
    def object_probe(self, *args, **kwargs):
        raise AttributeError("object probe unavailable")


# 中文注释：封装 _sample 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _sample(sample_id: str) -> Sample:
    return Sample(sample_id=sample_id, image=np.zeros((8, 8, 3), dtype=np.float32), text="")


# 中文注释：验证 test_caption_spr_is_neutral_when_no_clean_non_target_objects_present 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_caption_spr_is_neutral_when_no_clean_non_target_objects_present():
    cfg = AppConfig()
    cfg.task.object_probe_enabled = True
    row = {
        "target_object": "dog",
        "target_aliases": ["poodle"],
        "non_target_objects": ["cat"],
        "attack_goal": "remove_object",
    }

    metrics = _caption_metrics(
        row,
        NoProbeModel(),
        _sample("clean"),
        _sample("attacked"),
        _sample("defended"),
        ModelOutput(text="A brown poodle is sleeping on a shoe rack."),
        ModelOutput(text="A brown poodle is sleeping on a shoe rack."),
        ModelOutput(text="A brown poodle is sleeping on a wire rack."),
        cfg,
    )

    assert metrics["clean_non_target_present"] == []
    assert metrics["semantic_preservation_rate"] == 1.0
    assert metrics["object_jaccard"] == 1.0


# 中文注释：验证 test_vqa_case_bundle_scores_each_stage_by_answer_correctness 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_vqa_case_bundle_scores_each_stage_by_answer_correctness() -> None:
    cfg = AppConfig()
    cfg.task.kind = "vqa"
    cfg.plugins.model_adapter = "openai_qwen35_9b"
    sample = _sample("vqa-1")
    row = {"question": "What color are the shoes?", "answer": "white"}
    metrics = {
        "clean_correct": True,
        "attacked_correct": True,
        "defended_correct": True,
        "attack_success": False,
        "defense_recovered": False,
    }

    bundle = _case_bundle(
        cfg=cfg,
        row=row,
        clean_sample=sample,
        attacked_sample=sample,
        defended_sample=sample,
        clean_output=ModelOutput(text="white"),
        attacked_output=ModelOutput(text="white"),
        defended_output=ModelOutput(text="white"),
        stage_metrics=metrics,
        refs={},
        perturbation={"l2": 0.0, "linf": 0.0},
    )

    assert bundle["outputs"]["clean"]["score"] == 1.0
    assert bundle["outputs"]["adv"]["score"] == 1.0
    assert bundle["outputs"]["defended"]["score"] == 1.0


# 中文注释：验证 test_caption_case_bundle_scores_target_state_relative_to_clean 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_caption_case_bundle_scores_target_state_relative_to_clean() -> None:
    cfg = AppConfig()
    cfg.task.kind = "caption"
    sample = _sample("caption-1")
    metrics = {
        "target_present_clean": True,
        "target_present_attacked": False,
        "target_present_defended": True,
        "attack_success": True,
        "defense_recovered": True,
    }

    bundle = _case_bundle(
        cfg=cfg,
        row={"target_object": "dog"},
        clean_sample=sample,
        attacked_sample=sample,
        defended_sample=sample,
        clean_output=ModelOutput(text="A dog is on the grass."),
        attacked_output=ModelOutput(text="An animal is on the grass."),
        defended_output=ModelOutput(text="A dog is on the grass."),
        stage_metrics=metrics,
        refs={},
        perturbation={"l2": 0.0, "linf": 0.0},
    )

    assert bundle["outputs"]["clean"]["score"] == 1.0
    assert bundle["outputs"]["adv"]["score"] == 0.0
    assert bundle["outputs"]["defended"]["score"] == 1.0


# 中文注释：验证 test_vqa_answer_change_uses_normalized_text 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_vqa_answer_change_uses_normalized_text() -> None:
    from mmsec_eval.runner.generation_runner import _vqa_metrics

    metrics = _vqa_metrics(
        {"answer": "white", "answer_aliases": ["white and black"]},
        ModelOutput(text="White."),
        ModelOutput(text="white"),
        ModelOutput(text="white"),
    )

    assert metrics["clean_correct"] is True
    assert metrics["attacked_correct"] is True
    assert metrics["answer_changed"] is False
