# 文件说明：该文件属于自动化测试，集中实现 test fixture vlm adapter 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.model_adapters.fixture_vlm_adapter import FixtureVLMAdapter
from mmsec_eval.types import Sample


# 执行 `样本` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _sample(stage: str) -> Sample:
    return Sample(
        sample_id="s1",
        image=np.zeros((8, 8, 3), dtype=np.float32),
        text="What is shown?",
        target_text="circle",
        metadata={
            "generation_stage": stage,
            "answer": "circle",
            "attacked_answer": "square",
            "defended_answer": "circle",
            "clean_caption": "A red circle is shown on a plain background.",
            "attacked_caption": "A red square is shown on a plain background.",
            "defended_caption": "A red circle is shown on a plain background.",
            "target_object": "circle",
            "target_aliases": ["round object"],
            "attack_goal": "remove_object",
        },
    )


# 验证 `fixture 视觉问答 stage outputs` 场景，防止相关行为在后续修改中退化。
def test_fixture_vqa_stage_outputs() -> None:
    adapter = FixtureVLMAdapter()
    assert adapter.generate_answer(_sample("clean"), "What is shown?").text == "circle"
    assert adapter.generate_answer(_sample("attacked"), "What is shown?").text == "square"
    assert adapter.generate_answer(_sample("defended"), "What is shown?").text == "circle"


# 验证 `fixture 图像描述 and 探测 outputs` 场景，防止相关行为在后续修改中退化。
def test_fixture_caption_and_probe_outputs() -> None:
    adapter = FixtureVLMAdapter()
    assert "circle" in adapter.generate_caption(_sample("clean")).text
    assert "square" in adapter.generate_caption(_sample("attacked")).text
    assert adapter.object_probe(_sample("clean"), "circle").text == "yes"
    assert adapter.object_probe(_sample("attacked"), "circle").text == "no"
    assert adapter.object_probe(_sample("defended"), "circle").text == "yes"
