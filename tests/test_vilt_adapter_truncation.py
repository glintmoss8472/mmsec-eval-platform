# 文件说明：该文件属于自动化测试，集中实现 test vilt adapter truncation 相关逻辑。
from __future__ import annotations

import torch

from mmsec_eval.model_adapters.vilt_itm_adapter import ViltITMAdapter, _pick_text_length_limit


# 中文注释：定义 _FakeProcessor 的结构化职责，作为自动化测试中状态、配置或行为的边界。
class _FakeProcessor:
    # 中文注释：封装 _FakeProcessor.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self.kwargs = {}

    # 中文注释：封装 _FakeProcessor.__call__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 4, 4), dtype=torch.float32),
            "pixel_mask": torch.ones((1, 4, 4), dtype=torch.long),
        }


# 中文注释：验证 test_pick_text_length_limit_prefers_bounded_smallest 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_pick_text_length_limit_prefers_bounded_smallest():
    assert _pick_text_length_limit(None, 1000000000, 40, 256) == 40
    assert _pick_text_length_limit(None, -1, 0) == 40


# 中文注释：验证 test_prepare_inputs_enforces_truncation 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_prepare_inputs_enforces_truncation():
    adapter = ViltITMAdapter.__new__(ViltITMAdapter)
    adapter._processor = _FakeProcessor()
    adapter._device = "cpu"
    adapter._max_text_length = 40

    images = torch.rand((1, 3, 4, 4), dtype=torch.float32)
    out = adapter._prepare_inputs_torch(images, ["word " * 80])

    assert adapter._processor.kwargs["truncation"] is True
    assert adapter._processor.kwargs["max_length"] == 40
    assert adapter._processor.kwargs["padding"] is True
    assert tuple(out["input_ids"].shape) == (1, 3)
