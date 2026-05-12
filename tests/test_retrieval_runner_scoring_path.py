from __future__ import annotations

import numpy as np

from mmsec_eval.retrieval.metrics import VLRIndex
from mmsec_eval.runner.retrieval_runner import _score_matrix, _uses_pairwise_scoring


class DualStreamWithPairScore:
    def __init__(self) -> None:
        self.pair_calls = 0

    def encode_images_batch(self, images):
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(images)]

    def encode_texts_batch(self, texts):
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(texts)]

    def score_pairs(self, pairs, batch_size=16):
        self.pair_calls += len(pairs)
        return np.zeros((len(pairs),), dtype=np.float32)


class PairwiseOnly:
    def __init__(self) -> None:
        self.pair_calls = 0

    def score_pairs(self, pairs, batch_size=16):
        self.pair_calls += len(pairs)
        return np.asarray([1.0 if text == "a" else 0.0 for _, text in pairs], dtype=np.float32)


def _index() -> VLRIndex:
    img = np.zeros((2, 2, 3), dtype=np.float32)
    return VLRIndex(
        images=[img, img],
        texts=["a", "b"],
        image_ids=["i0", "i1"],
        text_ids=["t0", "t1"],
        gt_img_idx=np.asarray([0, 1], dtype=np.int64),
        gt_txt_idxs=[[0], [1]],
    )


def test_dual_stream_adapter_prefers_embedding_matrix_over_pairwise_scoring() -> None:
    adapter = DualStreamWithPairScore()

    sim = _score_matrix(adapter, _index(), batch_size=1)

    assert adapter.pair_calls == 0
    np.testing.assert_allclose(sim, np.eye(2, dtype=np.float32))
    assert not _uses_pairwise_scoring(adapter)


def test_pairwise_only_adapter_still_uses_pairwise_scoring() -> None:
    adapter = PairwiseOnly()

    sim = _score_matrix(adapter, _index(), batch_size=1)

    assert adapter.pair_calls == 4
    np.testing.assert_allclose(sim, np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32))
    assert _uses_pairwise_scoring(adapter)
