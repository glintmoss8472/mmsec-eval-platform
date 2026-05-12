# 文件说明：该文件属于图文检索评估层，集中实现 metrics 相关逻辑。
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmsec_eval.types import Sample


# 中文注释：定义 VLRIndex 的结构化职责，作为图文检索评估层中状态、配置或行为的边界。
@dataclass(frozen=True)
class VLRIndex:
    # Unique images (grouped by source_image) and caption texts.
    images: list[np.ndarray]
    texts: list[str]

    # Stable ids (useful for reporting).
    image_ids: list[str]
    text_ids: list[str]

    # Ground-truth mappings:
    # - For each text i, gt_img_idx[i] is the matching image index.
    # - For each image j, gt_txt_idxs[j] is the list of matching text indices.
    gt_img_idx: np.ndarray
    gt_txt_idxs: list[list[int]]


# 中文注释：实现 build_vlr_index 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def build_vlr_index(samples: list[Sample]) -> VLRIndex:
    """Build retrieval index from a caption-per-sample dataset.

    Expected sample.metadata.source_image to group captions by image.
    Falls back to sample.sample_id when source_image is missing.
    """
    image_id_to_idx: dict[str, int] = {}
    images: list[np.ndarray] = []
    image_ids: list[str] = []

    texts: list[str] = []
    text_ids: list[str] = []
    gt_img: list[int] = []
    gt_txt: list[list[int]] = []

    for s in samples:
        key = str(s.metadata.get("source_image") or s.metadata.get("image_id") or s.sample_id)
        if key not in image_id_to_idx:
            image_id_to_idx[key] = len(images)
            images.append(np.asarray(s.image))
            image_ids.append(key)
            gt_txt.append([])

        img_idx = image_id_to_idx[key]
        txt_idx = len(texts)
        texts.append(str(s.text or ""))
        text_ids.append(str(s.sample_id))
        gt_img.append(img_idx)
        gt_txt[img_idx].append(txt_idx)

    return VLRIndex(
        images=images,
        texts=texts,
        image_ids=image_ids,
        text_ids=text_ids,
        gt_img_idx=np.asarray(gt_img, dtype=np.int64),
        gt_txt_idxs=gt_txt,
    )


# 中文注释：实现 l2_normalize 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


# 中文注释：实现 score_matrix_dual_stream 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def score_matrix_dual_stream(img_embs: np.ndarray, txt_embs: np.ndarray) -> np.ndarray:
    """Return similarity matrix sim[text, image]."""
    img = l2_normalize(img_embs, axis=-1)
    txt = l2_normalize(txt_embs, axis=-1)
    if txt.ndim != 2 or img.ndim != 2 or txt.shape[1] != img.shape[1]:
        raise ValueError(f"bad embedding shapes: txt={txt.shape}, img={img.shape}")
    return (txt @ img.T).astype(np.float32)


# 中文注释：封装 _safe_ks 的内部步骤，让图文检索评估层主流程保持清晰并隔离边界细节。
def _safe_ks(ks: list[int]) -> list[int]:
    out = sorted({int(k) for k in ks if int(k) > 0})
    return out


# 中文注释：封装 _topk_sorted_indices 的内部步骤，让图文检索评估层主流程保持清晰并隔离边界细节。
def _topk_sorted_indices(sim: np.ndarray, max_k: int) -> np.ndarray:
    """Return per-row top-k indices sorted by score desc."""
    if max_k <= 0:
        return np.zeros((sim.shape[0], 0), dtype=np.int64)
    idx = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    scores = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


# 中文注释：实现 recall_at_k_t2i 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def recall_at_k_t2i(sim_t2i: np.ndarray, gt_img_idx: np.ndarray, ks: list[int]) -> dict[int, float]:
    """Text->Image recall@k.

    sim_t2i: [num_text, num_image]
    gt_img_idx: [num_text]
    """
    sim = np.asarray(sim_t2i, dtype=np.float32)
    gt = np.asarray(gt_img_idx, dtype=np.int64)
    if sim.ndim != 2:
        raise ValueError("sim_t2i must be 2D")
    n_txt, n_img = sim.shape
    if gt.shape[0] != n_txt:
        raise ValueError("gt_img_idx length mismatch")
    if n_txt == 0 or n_img == 0:
        return {k: 0.0 for k in _safe_ks(ks)}

    ks2 = _safe_ks(ks)
    max_k = min(max(ks2), n_img)

    top_idx = _topk_sorted_indices(sim, max_k)
    hits = top_idx == gt[:, None]
    out: dict[int, float] = {}
    for k in ks2:
        k2 = min(k, max_k)
        ok = hits[:, :k2].any(axis=1).mean()
        out[int(k)] = float(ok)
    return out


# 中文注释：实现 recall_at_k_i2t 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def recall_at_k_i2t(sim_i2t: np.ndarray, gt_txt_idxs: list[list[int]], ks: list[int]) -> dict[int, float]:
    """Image->Text recall@k.

    sim_i2t: [num_image, num_text]
    gt_txt_idxs: list of gt text indices per image
    """
    sim = np.asarray(sim_i2t, dtype=np.float32)
    if sim.ndim != 2:
        raise ValueError("sim_i2t must be 2D")
    n_img, n_txt = sim.shape
    if len(gt_txt_idxs) != n_img:
        raise ValueError("gt_txt_idxs length mismatch")
    if n_img == 0 or n_txt == 0:
        return {k: 0.0 for k in _safe_ks(ks)}

    ks2 = _safe_ks(ks)
    max_k = min(max(ks2), n_txt)
    top_idx = _topk_sorted_indices(sim, max_k)

    out: dict[int, float] = {}
    for k in ks2:
        k2 = min(k, max_k)
        ok = 0
        for i in range(n_img):
            gt_set = set(int(x) for x in gt_txt_idxs[i])
            if not gt_set:
                continue
            if any(int(x) in gt_set for x in top_idx[i, :k2].tolist()):
                ok += 1
        out[int(k)] = float(ok / max(1, n_img))
    return out


# 中文注释：实现 mean_rank_t2i 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def mean_rank_t2i(sim_t2i: np.ndarray, gt_img_idx: np.ndarray) -> float:
    """Mean rank for Text->Image (1 is best)."""
    sim = np.asarray(sim_t2i, dtype=np.float32)
    gt = np.asarray(gt_img_idx, dtype=np.int64)
    if sim.ndim != 2:
        raise ValueError("sim_t2i must be 2D")
    n_txt, n_img = sim.shape
    if n_txt == 0 or n_img == 0:
        return 0.0
    if gt.shape[0] != n_txt:
        raise ValueError("gt_img_idx length mismatch")

    # Tie-aware deterministic rank: if all scores are equal, ranks follow the
    # candidate index order instead of incorrectly reporting every GT as rank 1.
    gt_sim = sim[np.arange(n_txt), gt]
    candidate_idx = np.arange(n_img, dtype=np.int64)[None, :]
    rank = (
        1
        + (sim > gt_sim[:, None]).sum(axis=1)
        + ((sim == gt_sim[:, None]) & (candidate_idx < gt[:, None])).sum(axis=1)
    )
    return float(rank.mean())


# 中文注释：实现 mean_rank_i2t 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def mean_rank_i2t(sim_i2t: np.ndarray, gt_txt_idxs: list[list[int]]) -> float:
    """Mean best rank for Image->Text (1 is best).

    When an image has multiple GT captions, we use the best (minimum) rank among them.
    """
    sim = np.asarray(sim_i2t, dtype=np.float32)
    if sim.ndim != 2:
        raise ValueError("sim_i2t must be 2D")
    n_img, n_txt = sim.shape
    if n_img == 0 or n_txt == 0:
        return 0.0
    if len(gt_txt_idxs) != n_img:
        raise ValueError("gt_txt_idxs length mismatch")

    ranks: list[int] = []
    for i in range(n_img):
        gt = [int(x) for x in gt_txt_idxs[i] if 0 <= int(x) < n_txt]
        if not gt:
            continue
        candidate_idx = np.arange(n_txt, dtype=np.int64)
        gt_ranks = []
        for gt_idx in gt:
            gt_sim = float(sim[i, gt_idx])
            rank = (
                1
                + int((sim[i] > gt_sim).sum())
                + int(((sim[i] == gt_sim) & (candidate_idx < int(gt_idx))).sum())
            )
            gt_ranks.append(rank)
        ranks.append(int(min(gt_ranks)))
    return float(np.mean(ranks)) if ranks else 0.0


# 中文注释：实现 compute_vlr_metrics 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def compute_vlr_metrics(sim_t2i: np.ndarray, index: VLRIndex, ks: list[int]) -> dict[str, float]:
    """Compute both directions' metrics from sim[text,image]."""
    ks2 = _safe_ks(ks)
    ir_r = recall_at_k_t2i(sim_t2i, index.gt_img_idx, ks2)
    tr_r = recall_at_k_i2t(sim_t2i.T, index.gt_txt_idxs, ks2)
    ir_mr = mean_rank_t2i(sim_t2i, index.gt_img_idx)
    tr_mr = mean_rank_i2t(sim_t2i.T, index.gt_txt_idxs)

    out: dict[str, float] = {
        "mean_rank_ir": float(ir_mr),
        "mean_rank_tr": float(tr_mr),
    }
    for k in ks2:
        out[f"ir_r@{k}"] = float(ir_r.get(k, 0.0))
        out[f"tr_r@{k}"] = float(tr_r.get(k, 0.0))
        # This is the stage failure rate. For attack reporting,
        # use conditional_attack_metrics so clean failures are not counted as
        # attack successes.
        out[f"ir_asr@{k}"] = float(1.0 - ir_r.get(k, 0.0))
        out[f"tr_asr@{k}"] = float(1.0 - tr_r.get(k, 0.0))
    return out


# 中文注释：实现 topk_indices_t2i 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def topk_indices_t2i(sim_t2i: np.ndarray, k: int) -> np.ndarray:
    """Return top-k image indices per text query."""
    sim = np.asarray(sim_t2i, dtype=np.float32)
    if sim.ndim != 2:
        raise ValueError("sim_t2i must be 2D")
    n_txt, n_img = sim.shape
    if n_txt == 0 or n_img == 0:
        return np.zeros((n_txt, 0), dtype=np.int64)
    k2 = max(1, min(int(k), n_img))
    idx = np.argpartition(-sim, kth=k2 - 1, axis=1)[:, :k2]
    # Sort the selected k indices by score descending for stability.
    scores = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


# 中文注释：实现 topk_indices_i2t 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def topk_indices_i2t(sim_i2t: np.ndarray, k: int) -> np.ndarray:
    """Return top-k text indices per image query."""
    sim = np.asarray(sim_i2t, dtype=np.float32)
    if sim.ndim != 2:
        raise ValueError("sim_i2t must be 2D")
    n_img, n_txt = sim.shape
    if n_img == 0 or n_txt == 0:
        return np.zeros((n_img, 0), dtype=np.int64)
    k2 = max(1, min(int(k), n_txt))
    idx = np.argpartition(-sim, kth=k2 - 1, axis=1)[:, :k2]
    scores = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


# 中文注释：封装 _i2t_hit_maps 的内部步骤，让图文检索评估层主流程保持清晰并隔离边界细节。
def _i2t_hit_maps(
    *,
    clean_top: np.ndarray,
    attacked_top: np.ndarray,
    gt_txt_idxs: list[list[int]],
    ks: list[int],
    max_k_txt: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    clean_by_k: dict[int, np.ndarray] = {}
    attacked_by_k: dict[int, np.ndarray] = {}
    for k in ks:
        k_txt = min(int(k), max_k_txt)
        c_hits: list[bool] = []
        a_hits: list[bool] = []
        for img_idx, gt_texts in enumerate(gt_txt_idxs):
            gt_set = {int(x) for x in gt_texts}
            c_hits.append(any(int(x) in gt_set for x in clean_top[img_idx, :k_txt].tolist()))
            a_hits.append(any(int(x) in gt_set for x in attacked_top[img_idx, :k_txt].tolist()))
        clean_by_k[int(k)] = np.asarray(c_hits, dtype=bool)
        attacked_by_k[int(k)] = np.asarray(a_hits, dtype=bool)
    return clean_by_k, attacked_by_k


# 中文注释：封装 _add_conditional_success_metrics 的内部步骤，让图文检索评估层主流程保持清晰并隔离边界细节。
def _add_conditional_success_metrics(
    out: dict[str, float],
    *,
    ks: list[int],
    max_k_img: int,
    clean_t2i_hits: np.ndarray,
    attacked_t2i_hits: np.ndarray,
    clean_i2t_hit_by_k: dict[int, np.ndarray],
    attacked_i2t_hit_by_k: dict[int, np.ndarray],
) -> None:
    for k in ks:
        k_img = min(int(k), max_k_img)
        clean_ok = clean_t2i_hits[:, :k_img].any(axis=1)
        attacked_ok = attacked_t2i_hits[:, :k_img].any(axis=1)
        support = int(clean_ok.sum())
        success = clean_ok & ~attacked_ok
        out[f"ir_cond_asr@{k}"] = float(success.sum() / support) if support else 0.0
        out[f"ir_cond_support@{k}"] = float(support)
        out[f"ir_cond_success@{k}"] = float(success.sum())

        clean_ok_i = clean_i2t_hit_by_k[int(k)]
        attacked_ok_i = attacked_i2t_hit_by_k[int(k)]
        support_i = int(clean_ok_i.sum())
        success_i = clean_ok_i & ~attacked_ok_i
        out[f"tr_cond_asr@{k}"] = float(success_i.sum() / support_i) if support_i else 0.0
        out[f"tr_cond_support@{k}"] = float(support_i)
        out[f"tr_cond_success@{k}"] = float(success_i.sum())


# 中文注释：封装 _add_conditional_rank_metrics 的内部步骤，让图文检索评估层主流程保持清晰并隔离边界细节。
def _add_conditional_rank_metrics(out: dict[str, float], *, clean: np.ndarray, attacked: np.ndarray, index: VLRIndex, gt_img: np.ndarray) -> None:
    clean_gt = clean[np.arange(clean.shape[0]), gt_img]
    attacked_gt = attacked[np.arange(attacked.shape[0]), gt_img]
    clean_rank = 1 + (clean > clean_gt[:, None]).sum(axis=1)
    attacked_rank = 1 + (attacked > attacked_gt[:, None]).sum(axis=1)
    out["ir_rank_delta_mean"] = float(np.mean(attacked_rank - clean_rank))

    i2t_deltas: list[float] = []
    for img_idx, gt_texts in enumerate(index.gt_txt_idxs):
        gt = [int(x) for x in gt_texts if 0 <= int(x) < clean.shape[0]]
        if not gt:
            continue
        clean_best = float(np.max(clean.T[img_idx, gt]))
        attacked_best = float(np.max(attacked.T[img_idx, gt]))
        clean_rank_i = 1 + int((clean.T[img_idx] > clean_best).sum())
        attacked_rank_i = 1 + int((attacked.T[img_idx] > attacked_best).sum())
        i2t_deltas.append(float(attacked_rank_i - clean_rank_i))
    out["tr_rank_delta_mean"] = float(np.mean(i2t_deltas)) if i2t_deltas else 0.0


# 中文注释：实现 conditional_attack_metrics 的核心流程，支撑图文检索评估层中的业务语义和异常边界。
def conditional_attack_metrics(
    clean_sim_t2i: np.ndarray,
    attacked_sim_t2i: np.ndarray,
    index: VLRIndex,
    ks: list[int],
) -> dict[str, float]:
    """Measure attack success only on queries that were correct before attack.

    The usual stage failure rate over all samples overstates attack success when
    the clean model already failed. Conditional ASR counts a success only when
    the ground-truth item is in the clean top-k set and drops out after the attack.
    """
    clean = np.asarray(clean_sim_t2i, dtype=np.float32)
    attacked = np.asarray(attacked_sim_t2i, dtype=np.float32)
    if clean.shape != attacked.shape:
        raise ValueError(f"clean/attacked similarity shape mismatch: {clean.shape} vs {attacked.shape}")
    if clean.ndim != 2:
        raise ValueError("similarity matrices must be 2D")

    ks2 = _safe_ks(ks)
    out: dict[str, float] = {}
    if not ks2 or clean.shape[0] == 0 or clean.shape[1] == 0:
        return out

    max_k_img = min(max(ks2), clean.shape[1])
    max_k_txt = min(max(ks2), clean.shape[0])

    clean_t2i_top = topk_indices_t2i(clean, max_k_img)
    attacked_t2i_top = topk_indices_t2i(attacked, max_k_img)
    gt_img = np.asarray(index.gt_img_idx, dtype=np.int64)
    clean_t2i_hits = clean_t2i_top == gt_img[:, None]
    attacked_t2i_hits = attacked_t2i_top == gt_img[:, None]

    clean_i2t_top = topk_indices_i2t(clean.T, max_k_txt)
    attacked_i2t_top = topk_indices_i2t(attacked.T, max_k_txt)
    clean_i2t_hit_by_k, attacked_i2t_hit_by_k = _i2t_hit_maps(
        clean_top=clean_i2t_top,
        attacked_top=attacked_i2t_top,
        gt_txt_idxs=index.gt_txt_idxs,
        ks=ks2,
        max_k_txt=max_k_txt,
    )
    _add_conditional_success_metrics(
        out,
        ks=ks2,
        max_k_img=max_k_img,
        clean_t2i_hits=clean_t2i_hits,
        attacked_t2i_hits=attacked_t2i_hits,
        clean_i2t_hit_by_k=clean_i2t_hit_by_k,
        attacked_i2t_hit_by_k=attacked_i2t_hit_by_k,
    )
    _add_conditional_rank_metrics(out, clean=clean, attacked=attacked, index=index, gt_img=gt_img)
    return out
