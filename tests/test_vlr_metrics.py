# 文件说明：该文件属于自动化测试，集中实现 test vlr metrics 相关逻辑。
from __future__ import annotations

import numpy as np

from mmsec_eval.retrieval.metrics import (
    VLRIndex,
    build_vlr_index,
    conditional_attack_metrics,
    compute_vlr_metrics,
    recall_at_k_i2t,
    recall_at_k_t2i,
    score_matrix_dual_stream,
)
from mmsec_eval.types import Sample


# 中文注释：验证 test_vlr_metrics_perfect_retrieval 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_vlr_metrics_perfect_retrieval():
    # Two images, two captions each.
    img0 = np.zeros((4, 4, 3), dtype=np.float32)
    img1 = np.ones((4, 4, 3), dtype=np.float32)

    samples = [
        Sample(sample_id="t0", image=img0, text="cap0", metadata={"source_image": "i0"}),
        Sample(sample_id="t1", image=img0, text="cap1", metadata={"source_image": "i0"}),
        Sample(sample_id="t2", image=img1, text="cap2", metadata={"source_image": "i1"}),
        Sample(sample_id="t3", image=img1, text="cap3", metadata={"source_image": "i1"}),
    ]
    index = build_vlr_index(samples)
    assert isinstance(index, VLRIndex)
    assert len(index.images) == 2
    assert len(index.texts) == 4

    # Make embeddings align perfectly with GT mapping.
    img_embs = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # [I=2, D=2]
    txt_embs = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )  # [T=4, D=2]
    sim = score_matrix_dual_stream(img_embs, txt_embs)
    metrics = compute_vlr_metrics(sim, index=index, ks=[1, 5, 10])

    assert metrics["ir_r@1"] == 1.0
    assert metrics["tr_r@1"] == 1.0
    assert metrics["ir_asr@1"] == 0.0
    assert metrics["tr_asr@1"] == 0.0
    assert metrics["mean_rank_ir"] == 1.0
    assert metrics["mean_rank_tr"] == 1.0


# 中文注释：封装 _bruteforce_recall_t2i 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _bruteforce_recall_t2i(sim_t2i: np.ndarray, gt_img_idx: np.ndarray, k: int) -> float:
    sim = np.asarray(sim_t2i, dtype=np.float32)
    gt = np.asarray(gt_img_idx, dtype=np.int64)
    hit = 0
    for i in range(sim.shape[0]):
        rank = np.argsort(-sim[i])[:k]
        hit += int(int(gt[i]) in rank.tolist())
    return float(hit / max(1, sim.shape[0]))


# 中文注释：封装 _bruteforce_recall_i2t 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _bruteforce_recall_i2t(sim_i2t: np.ndarray, gt_txt_idxs: list[list[int]], k: int) -> float:
    sim = np.asarray(sim_i2t, dtype=np.float32)
    hit = 0
    for i in range(sim.shape[0]):
        rank = np.argsort(-sim[i])[:k].tolist()
        gt = set(int(x) for x in gt_txt_idxs[i])
        hit += int(any(int(x) in gt for x in rank))
    return float(hit / max(1, sim.shape[0]))


# 中文注释：验证 test_vlr_recall_matches_bruteforce 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_vlr_recall_matches_bruteforce():
    rng = np.random.default_rng(7)
    sim_t2i = rng.normal(size=(11, 7)).astype(np.float32)
    gt_img_idx = rng.integers(0, 7, size=(11,), endpoint=False).astype(np.int64)

    ks = [1, 3, 5]
    got_t2i = recall_at_k_t2i(sim_t2i, gt_img_idx, ks=ks)
    for k in ks:
        assert abs(got_t2i[k] - _bruteforce_recall_t2i(sim_t2i, gt_img_idx, k)) < 1e-9

    sim_i2t = rng.normal(size=(7, 11)).astype(np.float32)
    gt_txt_idxs: list[list[int]] = []
    for _ in range(7):
        ids = sorted(set(int(x) for x in rng.integers(0, 11, size=2, endpoint=False).tolist()))
        gt_txt_idxs.append(ids if ids else [0])
    got_i2t = recall_at_k_i2t(sim_i2t, gt_txt_idxs, ks=ks)
    for k in ks:
        assert abs(got_i2t[k] - _bruteforce_recall_i2t(sim_i2t, gt_txt_idxs, k)) < 1e-9


# 中文注释：验证 test_conditional_attack_metrics_only_count_clean_successes 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_conditional_attack_metrics_only_count_clean_successes():
    img0 = np.zeros((4, 4, 3), dtype=np.float32)
    img1 = np.ones((4, 4, 3), dtype=np.float32)
    samples = [
        Sample(sample_id="t0", image=img0, text="cap0", metadata={"source_image": "i0"}),
        Sample(sample_id="t1", image=img0, text="cap1", metadata={"source_image": "i0"}),
        Sample(sample_id="t2", image=img1, text="cap2", metadata={"source_image": "i1"}),
        Sample(sample_id="t3", image=img1, text="cap3", metadata={"source_image": "i1"}),
    ]
    index = build_vlr_index(samples)
    clean = np.asarray(
        [
            [5.0, 0.0],
            [4.0, 0.0],
            [0.0, 5.0],
            [0.0, 4.0],
        ],
        dtype=np.float32,
    )
    attacked = np.asarray(
        [
            [4.0, 5.0],
            [3.0, 4.0],
            [5.0, 4.5],
            [3.5, 3.0],
        ],
        dtype=np.float32,
    )

    got = conditional_attack_metrics(clean, attacked, index, ks=[1, 2])

    assert got["ir_cond_support@1"] == 4.0
    assert got["ir_cond_success@1"] == 4.0
    assert got["ir_cond_asr@1"] == 1.0
    assert got["tr_cond_support@1"] == 2.0
    assert got["tr_cond_success@1"] == 2.0
    assert got["tr_cond_asr@1"] == 1.0
    assert got["ir_cond_asr@2"] == 0.0
    assert got["tr_cond_asr@2"] == 0.0
    assert got["ir_rank_delta_mean"] > 0.0
    assert got["tr_rank_delta_mean"] > 0.0


# 中文注释：验证 test_mean_rank_is_tie_aware_for_flat_scores 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_mean_rank_is_tie_aware_for_flat_scores():
    img0 = np.zeros((4, 4, 3), dtype=np.float32)
    img1 = np.ones((4, 4, 3), dtype=np.float32)
    samples = [
        Sample(sample_id="t0", image=img0, text="cap0", metadata={"source_image": "i0"}),
        Sample(sample_id="t1", image=img1, text="cap1", metadata={"source_image": "i1"}),
    ]
    index = build_vlr_index(samples)
    sim = np.zeros((2, 2), dtype=np.float32)

    metrics = compute_vlr_metrics(sim, index=index, ks=[1])

    assert metrics["ir_r@1"] == 0.5
    assert metrics["tr_r@1"] == 0.5
    assert metrics["mean_rank_ir"] == 1.5
    assert metrics["mean_rank_tr"] == 1.5
