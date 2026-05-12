# 文件说明：该文件属于自动化测试，集中实现 test retrieval runner progress 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.retrieval.metrics import VLRIndex
from mmsec_eval.runner.retrieval_runner import _score_matrix


# 计算 `pairs`，为指标、风险或调度决策提供数值依据。
class _DummyCrossEncoder:
    # 实现 _DummyCrossEncoder.score_pairs 的核心行为，维护自动化测试在该对象上的调用契约。
    def score_pairs(self, pairs, batch_size: int = 1):  # noqa: ANN001
        del batch_size
        return np.asarray([0.5 for _ in pairs], dtype=np.float32)


# 验证 `分数 矩阵 报告 pair 进度` 场景，防止相关行为在后续修改中退化。
def test_score_matrix_reports_pair_progress():
    index = VLRIndex(
        images=[np.zeros((8, 8, 3), dtype=np.float32) for _ in range(2)],
        texts=["a", "b", "c"],
        image_ids=["i0", "i1"],
        text_ids=["t0", "t1", "t2"],
        gt_img_idx=np.asarray([0, 1, 0], dtype=np.int64),
        gt_txt_idxs={0: [0, 2], 1: [1]},
    )
    steps: list[tuple[int, int]] = []

    sim = _score_matrix(
        _DummyCrossEncoder(),
        index,
        batch_size=2,
        pair_progress=lambda done, total: steps.append((done, total)),
    )

    assert sim.shape == (3, 2)
    assert steps[-1] == (6, 6)
    assert steps == [(2, 6), (4, 6), (6, 6)]
