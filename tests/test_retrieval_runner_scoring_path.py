# 文件说明：该文件属于自动化测试，集中实现 test retrieval runner scoring path 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.retrieval.metrics import VLRIndex
from mmsec_eval.runner.retrieval_runner import _score_matrix, _uses_pairwise_scoring


# 实现 `DualStreamWithPairScore.__init__` 的对象行为，维护该类在自动化测试中的调用契约。
class DualStreamWithPairScore:
    # 封装 DualStreamWithPairScore.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self.pair_calls = 0

    # 实现 `DualStreamWithPairScore.encode_images_batch` 的对象行为，维护该类在自动化测试中的调用契约。
    def encode_images_batch(self, images):
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(images)]

    # 实现 `DualStreamWithPairScore.encode_texts_batch` 的对象行为，维护该类在自动化测试中的调用契约。
    def encode_texts_batch(self, texts):
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[: len(texts)]

    # 计算 `pairs`，为指标、风险或调度决策提供数值依据。
    def score_pairs(self, pairs, batch_size=16):
        self.pair_calls += len(pairs)
        return np.zeros((len(pairs),), dtype=np.float32)


# 实现 `PairwiseOnly.__init__` 的对象行为，维护该类在自动化测试中的调用契约。
class PairwiseOnly:
    # 封装 PairwiseOnly.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self.pair_calls = 0

    # 计算 `pairs`，为指标、风险或调度决策提供数值依据。
    def score_pairs(self, pairs, batch_size=16):
        self.pair_calls += len(pairs)
        return np.asarray([1.0 if text == "a" else 0.0 for _, text in pairs], dtype=np.float32)


# 执行 `索引` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
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


# 验证 `dual stream adapter prefers embedding 矩阵 over pairwise 评分` 场景，防止相关行为在后续修改中退化。
def test_dual_stream_adapter_prefers_embedding_matrix_over_pairwise_scoring() -> None:
    adapter = DualStreamWithPairScore()

    sim = _score_matrix(adapter, _index(), batch_size=1)

    assert adapter.pair_calls == 0
    np.testing.assert_allclose(sim, np.eye(2, dtype=np.float32))
    assert not _uses_pairwise_scoring(adapter)


# 验证 `pairwise only adapter still uses pairwise 评分` 场景，防止相关行为在后续修改中退化。
def test_pairwise_only_adapter_still_uses_pairwise_scoring() -> None:
    adapter = PairwiseOnly()

    sim = _score_matrix(adapter, _index(), batch_size=1)

    assert adapter.pair_calls == 4
    np.testing.assert_allclose(sim, np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32))
    assert _uses_pairwise_scoring(adapter)
