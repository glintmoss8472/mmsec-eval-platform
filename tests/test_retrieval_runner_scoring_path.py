# 文件说明：该文件属于自动化测试，集中实现 test retrieval runner scoring path 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.retrieval.metrics import VLRIndex
from mmsec_eval.runner.retrieval_runner import _score_matrix, _uses_pairwise_scoring


# 中文注释：定义 DualStreamWithPairScore 的结构化职责，作为自动化测试中状态、配置或行为的边界。
class DualStreamWithPairScore:
    # 中文注释：封装 DualStreamWithPairScore.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self.pair_calls = 0

    # 中文注释：实现 DualStreamWithPairScore.encode_images_batch 的核心行为，维护自动化测试在该对象上的调用契约。
    def encode_images_batch(self, images):
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(images)]

    # 中文注释：实现 DualStreamWithPairScore.encode_texts_batch 的核心行为，维护自动化测试在该对象上的调用契约。
    def encode_texts_batch(self, texts):
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(texts)]

    # 中文注释：实现 DualStreamWithPairScore.score_pairs 的核心行为，维护自动化测试在该对象上的调用契约。
    def score_pairs(self, pairs, batch_size=16):
        self.pair_calls += len(pairs)
        return np.zeros((len(pairs),), dtype=np.float32)


# 中文注释：定义 PairwiseOnly 的结构化职责，作为自动化测试中状态、配置或行为的边界。
class PairwiseOnly:
    # 中文注释：封装 PairwiseOnly.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self.pair_calls = 0

    # 中文注释：实现 PairwiseOnly.score_pairs 的核心行为，维护自动化测试在该对象上的调用契约。
    def score_pairs(self, pairs, batch_size=16):
        self.pair_calls += len(pairs)
        return np.asarray([1.0 if text == "a" else 0.0 for _, text in pairs], dtype=np.float32)


# 中文注释：封装 _index 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
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


# 中文注释：验证 test_dual_stream_adapter_prefers_embedding_matrix_over_pairwise_scoring 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_dual_stream_adapter_prefers_embedding_matrix_over_pairwise_scoring() -> None:
    adapter = DualStreamWithPairScore()

    sim = _score_matrix(adapter, _index(), batch_size=1)

    assert adapter.pair_calls == 0
    np.testing.assert_allclose(sim, np.eye(2, dtype=np.float32))
    assert not _uses_pairwise_scoring(adapter)


# 中文注释：验证 test_pairwise_only_adapter_still_uses_pairwise_scoring 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_pairwise_only_adapter_still_uses_pairwise_scoring() -> None:
    adapter = PairwiseOnly()

    sim = _score_matrix(adapter, _index(), batch_size=1)

    assert adapter.pair_calls == 4
    np.testing.assert_allclose(sim, np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32))
    assert _uses_pairwise_scoring(adapter)
