from __future__ import annotations

import torch

from mmsec_eval.model_adapters.vilt_itm_adapter import ViltITMAdapter, _pick_text_length_limit


class _FakeProcessor:
    def __init__(self) -> None:
        self.kwargs = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 4, 4), dtype=torch.float32),
            "pixel_mask": torch.ones((1, 4, 4), dtype=torch.long),
        }


def test_pick_text_length_limit_prefers_bounded_smallest():
    assert _pick_text_length_limit(None, 1000000000, 40, 256) == 40
    assert _pick_text_length_limit(None, -1, 0) == 40


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
