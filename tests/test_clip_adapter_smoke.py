from __future__ import annotations

import numpy as np

from mmsec_eval.model_adapters.clip_hf_adapter import ClipHFAdapter
from mmsec_eval.types import Sample


def test_clip_adapter_smoke():
    adapter = ClipHFAdapter()
    sample = Sample(sample_id="x", image=np.zeros((64, 64, 3), dtype=np.float32), text="a red circle")
    out = adapter.predict(sample)
    assert isinstance(out.score, float)
    assert out.embedding is not None
    assert out.text_embedding is not None
