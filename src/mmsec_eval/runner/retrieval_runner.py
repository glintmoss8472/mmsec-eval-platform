# 文件说明：该文件属于评测运行器，集中实现 retrieval runner 相关逻辑。
from __future__ import annotations

import logging
import re
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.datasets.registry import load_dataset
from mmsec_eval.plugins.registry import create
from mmsec_eval.model_adapters.local_vlm_lifecycle import (
    empty_cuda_cache,
    ensure_local_vlm_adapters_ready,
    local_vlm_adapters,
    stop_local_vlm_servers,
)
from mmsec_eval.retrieval.metrics import (
    VLRIndex,
    build_vlr_index,
    conditional_attack_metrics,
    compute_vlr_metrics,
    l2_normalize,
    score_matrix_dual_stream,
    topk_indices_t2i,
)
from mmsec_eval.risk.scoring import compute_risk_score, normalize_direct, normalize_inverse
from mmsec_eval.runner.artifacts import (
    make_run_dir,
    new_run_id,
    write_env_snapshot,
    write_json_snapshot,
    write_results,
    write_summary,
)
from mmsec_eval.runner.report import write_report
from mmsec_eval.sample_store.serializer import save_image_png, write_json, write_jsonl
from mmsec_eval.types import AttackContext, DefenseContext, RunArtifacts, Sample
from mmsec_eval.utils.seed import set_seed
from mmsec_eval.viz.plots import plot_defense_recovery_curve, plot_grouped_bar, plot_metric_curve, plot_stage_compare_bar

LOG = logging.getLogger(__name__)


# 发送 `进度` 回调或事件，让调用方及时感知运行状态。
def _emit_progress(progress: Callable[[str, str, float | None, str], None] | None, stage_key: str, state: str, progress_percent: float | None, message: str) -> None:
    if progress is not None:
        progress(stage_key, state, progress_percent, message)


# 安全计算 `均值`，在空值或异常输入下返回可控结果。
def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


# 计算 `矩阵 diagnostics`，为指标、风险或调度决策提供数值依据。
def _score_matrix_diagnostics(sim: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(sim, dtype=np.float32)
    if arr.size == 0:
        return {
            "available": False,
            "shape": list(arr.shape),
            "finite_ratio": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "unique_rounded": 0,
            "constant_score": True,
            "reason": "empty score matrix",
        }

    finite = np.isfinite(arr)
    finite_vals = arr[finite]
    if finite_vals.size == 0:
        return {
            "available": True,
            "shape": list(arr.shape),
            "finite_ratio": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "unique_rounded": 0,
            "constant_score": True,
            "reason": "no finite score values",
        }

    rounded = np.round(finite_vals.astype(np.float64), 6)
    std = float(np.std(finite_vals))
    unique = int(np.unique(rounded).size)
    constant = bool(std <= 1e-8 or unique <= 1)
    return {
        "available": True,
        "shape": list(arr.shape),
        "finite_ratio": float(finite_vals.size / max(1, arr.size)),
        "std": std,
        "min": float(np.min(finite_vals)),
        "max": float(np.max(finite_vals)),
        "unique_rounded": unique,
        "constant_score": constant,
        "reason": "constant or tied scoring" if constant else "score distribution has variation",
    }


# 执行 `victim stage ASR at 1` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _victim_stage_asr_at_1(metrics: dict[str, Any]) -> float:
    return 0.5 * (
        float(metrics.get("ir_asr@1", 0.0) or 0.0)
        + float(metrics.get("tr_asr@1", 0.0) or 0.0)
    )


# 执行 `victim conditional ASR at 1` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _victim_conditional_asr_at_1(metrics: dict[str, Any]) -> float:
    return 0.5 * (
        float(metrics.get("ir_cond_asr@1", 0.0) or 0.0)
        + float(metrics.get("tr_cond_asr@1", 0.0) or 0.0)
    )


# 计算 `指标 quality 报告`，把原始模型输出汇总成页面和报告使用的指标字段。
def _metric_quality_report(victim_metrics: dict[str, Any], victim_names: list[str]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    for victim_name in victim_names:
        node = victim_metrics.get(victim_name, {}) if isinstance(victim_metrics.get(victim_name, {}), dict) else {}
        for stage in ("clean", "attacked", "defended_attack", "defended_clean"):
            metrics = node.get(stage, {}) if isinstance(node.get(stage, {}), dict) else {}
            diag = metrics.get("score_diagnostics", {}) if isinstance(metrics.get("score_diagnostics", {}), dict) else {}
            if not diag:
                continue
            if float(diag.get("finite_ratio", 1.0) or 0.0) < 1.0:
                flags.append(
                    {
                        "severity": "error",
                        "victim": victim_name,
                        "stage": stage,
                        "code": "non_finite_scores",
                        "message": "相似度矩阵包含非有限值，不能用于攻击强度结论。",
                        "diagnostics": diag,
                    }
                )
            if bool(diag.get("constant_score", False)):
                flags.append(
                    {
                        "severity": "warning",
                        "victim": victim_name,
                        "stage": stage,
                        "code": "constant_scores",
                        "message": "该阶段打分几乎为常数，Recall/错误率主要反映排序并列，不应解释为有效攻击成功率。",
                        "diagnostics": diag,
                    }
                )

        clean = node.get("clean", {}) if isinstance(node.get("clean", {}), dict) else {}
        attacked = node.get("attacked", {}) if isinstance(node.get("attacked", {}), dict) else {}
        conditional = node.get("conditional", {}) if isinstance(node.get("conditional", {}), dict) else {}
        clean_error = _victim_stage_asr_at_1(clean) if clean else 0.0
        attacked_error = _victim_stage_asr_at_1(attacked) if attacked else 0.0
        conditional_asr = _victim_conditional_asr_at_1(conditional) if conditional else 0.0
        if clean_error >= 0.9 and attacked_error >= 0.9 and conditional_asr <= 0.01:
            flags.append(
                {
                    "severity": "warning",
                    "victim": victim_name,
                    "stage": "attacked",
                    "code": "baseline_failure_not_attack_success",
                    "message": "clean 阶段已经大面积失败，攻击后错误率不能当作条件攻击成功率。",
                    "clean_error_rate@1": float(clean_error),
                    "attacked_error_rate@1": float(attacked_error),
                    "conditional_asr@1": float(conditional_asr),
                }
            )

    blocking_codes = {"non_finite_scores", "constant_scores"}
    return {
        "valid_for_attack_strength_claim": not any(str(flag.get("code")) in blocking_codes for flag in flags),
        "has_warnings": bool(flags),
        "flags": flags,
        "note": "攻击成功率使用“正常阶段首位命中、攻击后首位掉出”的条件口径；攻击后错误率仅作诊断。",
    }


# 执行 `victim transfer 分数` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _victim_transfer_score(
    victim_metrics: dict[str, dict[str, Any]],
    victim_names: list[str],
    *,
    threshold: float,
) -> tuple[float, list[float]]:
    victim_asrs: list[float] = []
    for victim_name in victim_names:
        node = victim_metrics.get(victim_name, {})
        conditional_metrics = node.get("conditional", {}) if isinstance(node, dict) else {}
        attacked_metrics = node.get("attacked", {}) if isinstance(node, dict) else {}
        if isinstance(conditional_metrics, dict) and conditional_metrics:
            victim_asrs.append(_victim_conditional_asr_at_1(conditional_metrics))
        elif isinstance(attacked_metrics, dict):
            victim_asrs.append(_victim_stage_asr_at_1(attacked_metrics))
    if len(victim_asrs) <= 1:
        return 0.0, victim_asrs
    hit_count = sum(1 for value in victim_asrs if value >= float(threshold))
    return float(hit_count / len(victim_asrs)), victim_asrs


# 整理 `sanitize 目录 名称` 路径信息，把本地文件或产物引用转换成统一表示。
def _sanitize_dir_name(name: str) -> str:
    # Keep it filesystem-safe and short.
    x = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name))
    return x[:120] if len(x) > 120 else x


# 筛选 `victim names`，按配置条件保留可用于评测或展示的数据。
def _select_victim_names(cfg: AppConfig) -> tuple[str, list[str]]:
    surrogate = str(cfg.runner.surrogate_model_adapter or cfg.plugins.model_adapter)
    victims = list(cfg.runner.victim_model_adapters or [])
    if not victims:
        victims = [str(cfg.plugins.model_adapter)]
    return surrogate, [str(x) for x in victims]


# 执行 `索引 downsample` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _index_downsample(index: VLRIndex, max_pairs: int, seed: int) -> VLRIndex:
    if max_pairs <= 0:
        return index
    m = len(index.images)
    n = len(index.texts)
    if m == 0 or n == 0:
        return index
    if m * n <= max_pairs:
        return index

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(m).tolist()

    selected_imgs: list[int] = []
    selected_texts: set[int] = set()

    for img_idx in order:
        cand_texts = set(int(t) for t in index.gt_txt_idxs[img_idx])
        new_texts = selected_texts | cand_texts
        new_pairs = (len(selected_imgs) + 1) * len(new_texts)
        if selected_imgs and new_pairs > max_pairs:
            continue
        selected_imgs.append(int(img_idx))
        selected_texts = new_texts
        if len(selected_imgs) * len(selected_texts) >= max_pairs:
            break

    if not selected_imgs or not selected_texts:
        # Fallback: take first image and its texts.
        selected_imgs = [0]
        selected_texts = set(index.gt_txt_idxs[0])

    selected_imgs = sorted(set(selected_imgs))
    selected_text_list = sorted(set(int(x) for x in selected_texts))

    # Remap to compact indices.
    img_old_to_new = {old: i for i, old in enumerate(selected_imgs)}
    txt_old_to_new = {old: i for i, old in enumerate(selected_text_list)}

    images = [index.images[i] for i in selected_imgs]
    image_ids = [index.image_ids[i] for i in selected_imgs]
    texts = [index.texts[i] for i in selected_text_list]
    text_ids = [index.text_ids[i] for i in selected_text_list]

    gt_img_idx = []
    for old_txt in selected_text_list:
        old_img = int(index.gt_img_idx[old_txt])
        gt_img_idx.append(int(img_old_to_new[old_img]))

    gt_txt_idxs: list[list[int]] = [[] for _ in range(len(images))]
    for old_img in selected_imgs:
        new_img = int(img_old_to_new[old_img])
        for old_txt in index.gt_txt_idxs[old_img]:
            if int(old_txt) in txt_old_to_new:
                gt_txt_idxs[new_img].append(int(txt_old_to_new[int(old_txt)]))

    return VLRIndex(
        images=images,
        texts=texts,
        image_ids=image_ids,
        text_ids=text_ids,
        gt_img_idx=np.asarray(gt_img_idx, dtype=np.int64),
        gt_txt_idxs=gt_txt_idxs,
    )


# 执行 `encode 图像` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _encode_images(adapter: Any, images: list[np.ndarray], *, batch_size: int = 16) -> np.ndarray:
    if hasattr(adapter, "encode_images_batch"):
        try:
            encoded = adapter.encode_images_batch(images, batch_size=max(1, int(batch_size)))
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            encoded = adapter.encode_images_batch(images)
        return np.asarray(encoded, dtype=np.float32)
    if hasattr(adapter, "encode_image"):
        arr = np.stack([np.asarray(adapter.encode_image(im), dtype=np.float32) for im in images], axis=0)
        return arr
    raise RuntimeError(
        "Retrieval evaluation requires a dual-stream adapter with encode_images_batch/encode_image. "
        f"Adapter {type(adapter).__name__} does not implement image encoding."
    )


# 执行 `encode texts` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _encode_texts(adapter: Any, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
    if hasattr(adapter, "encode_texts_batch"):
        try:
            encoded = adapter.encode_texts_batch(texts, batch_size=max(1, int(batch_size)))
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            encoded = adapter.encode_texts_batch(texts)
        return np.asarray(encoded, dtype=np.float32)
    if hasattr(adapter, "encode_text"):
        arr = np.stack([np.asarray(adapter.encode_text(t), dtype=np.float32) for t in texts], axis=0)
        return arr
    raise RuntimeError(
        "Retrieval evaluation requires a dual-stream adapter with encode_texts_batch/encode_text. "
        f"Adapter {type(adapter).__name__} does not implement text encoding."
    )


# 执行 `supports dual stream` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _supports_dual_stream(adapter: Any) -> bool:
    has_image_encoder = hasattr(adapter, "encode_images_batch") or hasattr(adapter, "encode_image")
    has_text_encoder = hasattr(adapter, "encode_texts_batch") or hasattr(adapter, "encode_text")
    return bool(has_image_encoder and has_text_encoder)


# 执行 `uses pairwise 评分` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _uses_pairwise_scoring(adapter: Any) -> bool:
    return bool(hasattr(adapter, "score_pairs") and not _supports_dual_stream(adapter))


# 规范化 `文本 similarity ratio` 字段，把空值和非字符串输入转换为稳定文本。
def _text_similarity_ratio(clean_texts: list[str], attacked_texts: list[str]) -> float | None:
    pairs = list(zip(clean_texts, attacked_texts))
    if not pairs:
        return None
    vals = [SequenceMatcher(None, str(a), str(b)).ratio() for a, b in pairs]
    return float(_safe_mean([float(x) for x in vals]))


# 计算 `paired cosine 均值` 均值，空输入时返回可控的默认结果。
def _paired_cosine_mean(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape or x.shape[0] == 0:
        return None
    xn = l2_normalize(x, axis=-1)
    yn = l2_normalize(y, axis=-1)
    vals = np.sum(xn * yn, axis=1)
    return float(np.clip(np.mean(vals), -1.0, 1.0))


# 执行 `semantic preservation` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _semantic_preservation(
    *,
    surrogate_adapter: Any,
    clean_index: VLRIndex,
    attacked_index: VLRIndex | None,
    avg_linf: float,
    cfg: AppConfig,
) -> dict[str, Any]:
    if attacked_index is None:
        return {"available": False, "reason": "no attacked index"}

    text_ratio = _text_similarity_ratio(clean_index.texts, attacked_index.texts)
    pixel_proxy = normalize_inverse(float(avg_linf), float(cfg.risk.linf_reference))
    image_cosine: float | None = None
    image_error = ""

    if _supports_dual_stream(surrogate_adapter) and len(clean_index.images) == len(attacked_index.images):
        try:
            clean_img_emb = _encode_images(surrogate_adapter, clean_index.images)
            adv_img_emb = _encode_images(surrogate_adapter, attacked_index.images)
            image_cosine = _paired_cosine_mean(clean_img_emb, adv_img_emb)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            image_error = str(exc)

    components = [
        x
        for x in (
            image_cosine,
            text_ratio,
            pixel_proxy,
        )
        if x is not None
    ]
    combined = float(_safe_mean([float(max(0.0, min(1.0, x))) for x in components]))
    payload: dict[str, Any] = {
        "available": bool(components),
        "combined_semantic_preservation": combined,
        "clip_image_image_similarity": None if image_cosine is None else float(image_cosine),
        "text_similarity": None if text_ratio is None else float(text_ratio),
        "pixel_linf_preservation_proxy": float(pixel_proxy),
        "num_image_pairs": int(min(len(clean_index.images), len(attacked_index.images))),
        "num_text_pairs": int(min(len(clean_index.texts), len(attacked_index.texts))),
        "method": "clip-image cosine + text SequenceMatcher + inverse Linf proxy",
    }
    if image_error:
        payload["image_embedding_error"] = image_error
    return payload


# 计算 `two prompts`，为指标、风险或调度决策提供数值依据。
def _score_two_prompts(adapter: Any, image: np.ndarray, present_prompt: str, absent_prompt: str) -> tuple[float, float] | None:
    try:
        if _supports_dual_stream(adapter):
            img_emb = _encode_images(adapter, [image])
            txt_emb = _encode_texts(adapter, [present_prompt, absent_prompt])
            sim = score_matrix_dual_stream(img_emb, txt_emb)
            return float(sim[0, 0]), float(sim[1, 0])
        if hasattr(adapter, "score_pairs"):
            scores = np.asarray(
                adapter.score_pairs([(image, present_prompt), (image, absent_prompt)], batch_size=2),
                dtype=np.float32,
            ).reshape(-1)
            if scores.shape[0] >= 2:
                return float(scores[0]), float(scores[1])
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return None


# 计算 `图像 文本`，为指标、风险或调度决策提供数值依据。
def _score_image_text(adapter: Any, image: np.ndarray, text: str) -> float | None:
    try:
        if _supports_dual_stream(adapter):
            img_emb = _encode_images(adapter, [image])
            txt_emb = _encode_texts(adapter, [text])
            sim = score_matrix_dual_stream(img_emb, txt_emb)
            return float(sim[0, 0])
        if hasattr(adapter, "score_pairs"):
            scores = np.asarray(adapter.score_pairs([(image, text)], batch_size=1), dtype=np.float32).reshape(-1)
            if scores.shape[0] >= 1:
                return float(scores[0])
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return None


# 整理 `object proxy rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _object_proxy_rows(clean_samples: list[Sample], attacked_samples: list[Sample], max_cases: int) -> list[tuple[Sample, Sample, str]]:
    attacked_by_id = {str(s.sample_id): s for s in attacked_samples}
    rows: list[tuple[Sample, Sample, str]] = []
    for clean in clean_samples:
        target = str(clean.target_text or clean.metadata.get("target_text") or clean.metadata.get("object_category") or "").strip()
        adv = attacked_by_id.get(str(clean.sample_id))
        if target and adv is not None:
            rows.append((clean, adv, target))
        if len(rows) >= max(1, int(max_cases)):
            break
    return rows


# 执行 `object proxy empty` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _object_proxy_empty(reason: str, note: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "note": note}


# 执行 `object decision proxy` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _object_decision_proxy(
    *,
    adapter: Any,
    clean_samples: list[Sample],
    attacked_samples: list[Sample],
    max_cases: int = 512,
) -> dict[str, Any]:
    rows = _object_proxy_rows(clean_samples, attacked_samples, max_cases)
    if not rows:
        return _object_proxy_empty(
            "no samples with target_text/object_category",
            "Prepare a COCO object subset to enable the AdvEDM-style object decision proxy.",
        )

    clean_present = 0
    adv_present = 0
    flips = 0
    valid_wrong = 0
    margins_clean: list[float] = []
    margins_adv: list[float] = []
    preview: list[dict[str, Any]] = []

    for clean, adv, target in rows:
        present_prompt = f"a photo containing {target}"
        absent_prompt = f"a photo without {target}"
        clean_scores = _score_two_prompts(adapter, np.asarray(clean.image, dtype=np.float32), present_prompt, absent_prompt)
        adv_scores = _score_two_prompts(adapter, np.asarray(adv.image, dtype=np.float32), present_prompt, absent_prompt)
        if clean_scores is None or adv_scores is None:
            continue
        clean_margin = float(clean_scores[0] - clean_scores[1])
        adv_margin = float(adv_scores[0] - adv_scores[1])
        clean_is_present = clean_margin >= 0.0
        adv_is_present = adv_margin >= 0.0
        clean_present += int(clean_is_present)
        adv_present += int(adv_is_present)
        flips += int(clean_is_present != adv_is_present)
        valid_wrong += int(clean_is_present and not adv_is_present)
        margins_clean.append(clean_margin)
        margins_adv.append(adv_margin)
        if len(preview) < 20:
            preview.append(
                {
                    "sample_id": str(clean.sample_id),
                    "target_text": target,
                    "clean_margin": clean_margin,
                    "attacked_margin": adv_margin,
                    "clean_present": bool(clean_is_present),
                    "attacked_present": bool(adv_is_present),
                    "valid_wrong": bool(clean_is_present and not adv_is_present),
                }
            )

    n = len(margins_clean)
    if n == 0:
        return _object_proxy_empty("adapter could not score target prompts", "Object decision proxy needs either dual-stream encoders or score_pairs.")
    return {
        "available": True,
        "num_cases": int(n),
        "clean_present_rate": float(clean_present / n),
        "attacked_present_rate": float(adv_present / n),
        "decision_flip_rate": float(flips / n),
        "valid_wrong_rate": float(valid_wrong / n),
        "mean_margin_clean": float(_safe_mean(margins_clean)),
        "mean_margin_attacked": float(_safe_mean(margins_adv)),
        "preview": preview,
        "note": (
            "This is a CLIP-style target-presence proxy for AdvEDM object-level analysis; "
            "it is not a full embodied simulator rollout."
        ),
    }


# 计算 `矩阵`，为指标、风险或调度决策提供数值依据。
def _score_matrix(
    adapter: Any,
    index: VLRIndex,
    *,
    batch_size: int = 16,
    pair_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Return sim[text, image] for either dual-stream or cross-encoder victims."""
    if _supports_dual_stream(adapter):
        # Dual-stream victims expose image/text encoders, so one similarity matrix is enough.
        img_embs = _encode_images(adapter, index.images, batch_size=batch_size)
        txt_embs = _encode_texts(adapter, index.texts, batch_size=max(batch_size, 2 * batch_size))
        return score_matrix_dual_stream(img_embs, txt_embs)

    if hasattr(adapter, "score_pairs"):
        # Cross-encoder victims do not expose separate image/text embeddings, so
        # we score every (text, image) pair explicitly and fill the matrix row by row.
        pairs: list[tuple[np.ndarray, str]] = []
        coords: list[tuple[int, int]] = []
        n_txt = len(index.texts)
        n_img = len(index.images)
        total_pairs = int(n_txt * n_img)
        completed_pairs = 0
        sim = np.zeros((n_txt, n_img), dtype=np.float32)
        for ti, text in enumerate(index.texts):
            for ii, img in enumerate(index.images):
                pairs.append((img, text))
                coords.append((ti, ii))
                if len(pairs) >= max(1, int(batch_size)):
                    scores = adapter.score_pairs(pairs, batch_size=max(1, int(batch_size)))
                    for (t_idx, i_idx), s in zip(coords, scores.tolist()):
                        sim[t_idx, i_idx] = float(s)
                    completed_pairs += len(coords)
                    if pair_progress is not None:
                        pair_progress(completed_pairs, total_pairs)
                    pairs.clear()
                    coords.clear()
        if pairs:
            scores = adapter.score_pairs(pairs, batch_size=max(1, int(batch_size)))
            for (t_idx, i_idx), s in zip(coords, scores.tolist()):
                sim[t_idx, i_idx] = float(s)
            completed_pairs += len(coords)
            if pair_progress is not None:
                pair_progress(completed_pairs, total_pairs)
        return sim

    raise RuntimeError(
        "Retrieval evaluation requires either dual-stream image/text encoders or pairwise scoring. "
        f"Adapter {type(adapter).__name__} implements neither usable path."
    )


# 判断 `是否需要 parallelize victims` 条件是否成立，为调用方提供布尔决策。
def _should_parallelize_victims(victim_names: list[str]) -> bool:
    names = [str(x) for x in victim_names]
    if len(names) <= 1:
        return False
    return all(name.startswith("openai_") for name in names)


# 评估 `victim stage` 结果，汇总攻击前后指标和风险证据。
def _evaluate_victim_stage(
    *,
    victims: dict[str, Any],
    victim_names: list[str],
    index: VLRIndex,
    ks: list[int],
    batch_size: int,
    progress: Callable[[str, str, float | None, str], None] | None = None,
    progress_start: float = 0.0,
    progress_end: float = 100.0,
    progress_message: str = "",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    victim_units = {
        victim_name: int(len(index.texts) * len(index.images))
        if _uses_pairwise_scoring(victims[victim_name])
        else 1
        for victim_name in victim_names
    }
    total_units = max(1, sum(int(x) for x in victim_units.values()))
    completed_units = 0

    # 发送 `stage 进度` 回调或事件，让调用方及时感知运行状态。
    def _emit_stage_progress(victim_name: str, done_units: int, total_for_victim: int) -> None:
        if progress is None:
            return
        overall_done = min(total_units, completed_units + max(0, int(done_units)))
        ratio = float(overall_done) / float(max(1, total_units))
        pct = float(progress_start) + (float(progress_end) - float(progress_start)) * ratio
        if total_for_victim > 1:
            message = (
                f"{progress_message}：{victim_name}，"
                f"已完成 {int(done_units)}/{int(total_for_victim)} 对图文配对。"
            )
        else:
            message = f"{progress_message}：{victim_name}。"
        _emit_progress(progress, "victim_evaluation", "running", pct, message)

    # 执行 `work` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
    def _work(victim_name: str) -> tuple[str, np.ndarray, dict[str, Any]]:
        adapter = victims[victim_name]
        total_for_victim = int(victim_units.get(victim_name, 1))
        if total_for_victim > 1:
            sim = _score_matrix(
                adapter,
                index,
                batch_size=batch_size,
                pair_progress=lambda done, total: _emit_stage_progress(victim_name, done, total),
            )
        else:
            _emit_stage_progress(victim_name, 0, total_for_victim)
            sim = _score_matrix(adapter, index, batch_size=batch_size)
        metrics = compute_vlr_metrics(sim, index=index, ks=ks)
        score_diag = _score_matrix_diagnostics(sim)
        metrics["score_diagnostics"] = score_diag
        metrics["score_std"] = float(score_diag.get("std", 0.0) or 0.0)
        metrics["score_unique_rounded"] = float(score_diag.get("unique_rounded", 0) or 0)
        metrics["score_constant"] = float(1.0 if bool(score_diag.get("constant_score", False)) else 0.0)
        return victim_name, sim, metrics

    sims: dict[str, np.ndarray] = {}
    metrics_by_victim: dict[str, Any] = {}
    if progress is None and _should_parallelize_victims(victim_names):
        with ThreadPoolExecutor(max_workers=len(victim_names)) as pool:
            futures = {pool.submit(_work, victim_name): victim_name for victim_name in victim_names}
            for fut in as_completed(futures):
                victim_name, sim, metrics = fut.result()
                sims[victim_name] = sim
                metrics_by_victim[victim_name] = metrics
        return sims, metrics_by_victim

    for victim_name in victim_names:
        victim_name, sim, metrics = _work(victim_name)
        sims[victim_name] = sim
        metrics_by_victim[victim_name] = metrics
        completed_units += int(victim_units.get(victim_name, 1))
        _emit_stage_progress(victim_name, int(victim_units.get(victim_name, 1)), int(victim_units.get(victim_name, 1)))
    return sims, metrics_by_victim


# 执行 `pca project 2d` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _pca_project_2d(vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        if vt.shape[0] == 0:
            return np.zeros((x.shape[0], 2), dtype=np.float32)
        basis = vt[:2].T
        z = x @ basis
    except (np.linalg.LinAlgError, ValueError):
        z = x[:, :2] if x.shape[1] >= 2 else np.pad(x[:, :1], ((0, 0), (0, 1)))
    if z.shape[1] < 2:
        z = np.pad(z, ((0, 0), (0, 2 - z.shape[1])))
    return z[:, :2].astype(np.float32)


# 执行 `样本 ids` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _sample_ids(n: int, max_n: int) -> list[int]:
    n = int(max(0, n))
    m = int(max(1, max_n))
    if n <= m:
        return list(range(n))
    return np.linspace(0, n - 1, num=m, dtype=np.int64).tolist()


# 执行 `调试 产物 enabled` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _debug_artifacts_enabled(cfg: AppConfig) -> bool:
    return bool(
        bool(getattr(cfg.sample_store, "enabled", False))
        or bool(getattr(cfg.report, "save_patch_preview", False))
        or bool(getattr(cfg.report, "save_heatmaps", False))
    )


# 构建 `feature projection` 数据，集中整理评测运行器需要的输出结构。
def _build_feature_projection(
    *,
    surrogate_adapter: Any,
    clean_index: VLRIndex,
    attacked_index: VLRIndex | None,
    defended_attack_index: VLRIndex | None,
    defended_clean_index: VLRIndex | None,
    max_points_per_group: int = 64,
) -> dict[str, Any]:
    if not (hasattr(surrogate_adapter, "encode_images_batch") and hasattr(surrogate_adapter, "encode_texts_batch")):
        return {"available": False, "reason": "surrogate adapter does not support dual-stream encoders", "points": []}

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []

    # 执行 `append stage` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
    def _append_stage(stage: str, idx: VLRIndex) -> None:
        if idx is None:
            return
        img_ids = _sample_ids(len(idx.images), max_points_per_group)
        txt_ids = _sample_ids(len(idx.texts), max_points_per_group)
        if img_ids:
            imgs = [idx.images[i] for i in img_ids]
            img_emb = np.asarray(surrogate_adapter.encode_images_batch(imgs), dtype=np.float32)
            for local_i, emb in enumerate(img_emb):
                vectors.append(np.asarray(emb, dtype=np.float32).reshape(-1))
                rows.append(
                    {
                        "stage": stage,
                        "modality": "image",
                        "id": str(idx.image_ids[img_ids[local_i]]),
                    }
                )
        if txt_ids:
            txts = [idx.texts[i] for i in txt_ids]
            txt_emb = np.asarray(surrogate_adapter.encode_texts_batch(txts), dtype=np.float32)
            for local_i, emb in enumerate(txt_emb):
                vectors.append(np.asarray(emb, dtype=np.float32).reshape(-1))
                rows.append(
                    {
                        "stage": stage,
                        "modality": "text",
                        "id": str(idx.text_ids[txt_ids[local_i]]),
                    }
                )

    try:
        _append_stage("clean", clean_index)
        if attacked_index is not None:
            _append_stage("attacked", attacked_index)
        if defended_attack_index is not None:
            _append_stage("defended_attack", defended_attack_index)
        if defended_clean_index is not None:
            _append_stage("defended_clean", defended_clean_index)

        if not vectors:
            return {"available": False, "reason": "no vectors", "points": []}

        min_dim = min(v.shape[0] for v in vectors)
        x = np.stack([v[:min_dim] for v in vectors], axis=0)
        proj = _pca_project_2d(x)

        points: list[dict[str, Any]] = []
        for i, meta in enumerate(rows):
            points.append(
                {
                    **meta,
                    "x": float(proj[i, 0]),
                    "y": float(proj[i, 1]),
                }
            )
        return {
            "available": True,
            "method": "pca",
            "num_points": len(points),
            "points": points,
        }
    except (IndexError, RuntimeError, TypeError, ValueError) as e:
        return {"available": False, "reason": str(e), "points": []}


# 推断 `攻击 作用范围 样本`，从样本、配置或运行记录中提取统一名称。
def _attack_scope_samples(
    *,
    cfg: AppConfig,
    run_dir: str,
    clean_samples: list[Sample],
    clean_index: VLRIndex,
    attack: Any,
    surrogate_adapter: Any,
    scope: str,
) -> tuple[list[Sample], dict[str, Any]]:
    """Create attacked samples while keeping the same image/text grouping as clean."""
    scope = str(scope)
    need_img = scope in {"image", "joint"}
    need_txt = scope in {"text", "joint"}
    cfg_img = cfg
    cfg_txt = cfg
    if scope == "joint":
        cfg_img = copy.deepcopy(cfg)
        cfg_img.task.eval_scope = "image"
        cfg_txt = copy.deepcopy(cfg)
        cfg_txt.task.eval_scope = "text"

    attack_debug_root = Path(run_dir) / "attack_debug"
    attack_debug_root.mkdir(parents=True, exist_ok=True)
    debug_enabled = _debug_artifacts_enabled(cfg)
    image_to_anchor, image_to_clean = _scope_image_maps(clean_samples)
    adv_images, adv_image_info, l2_values, linf_values = _attack_unique_images(
        cfg=cfg_img,
        run_dir=run_dir,
        attack_debug_root=attack_debug_root,
        attack=attack,
        surrogate_adapter=surrogate_adapter,
        image_to_anchor=image_to_anchor,
        image_to_clean=image_to_clean,
        debug_enabled=debug_enabled,
    ) if need_img else ({}, {}, [], [])
    adv_texts, text_changed_flags = _attack_scope_texts(
        cfg=cfg_txt,
        run_dir=run_dir,
        attack_debug_root=attack_debug_root,
        clean_samples=clean_samples,
        attack=attack,
        surrogate_adapter=surrogate_adapter,
        debug_enabled=debug_enabled,
    ) if need_txt else ({}, [])
    attacked_samples = _assemble_attacked_scope_samples(
        cfg=cfg,
        clean_samples=clean_samples,
        scope=scope,
        need_img=need_img,
        need_txt=need_txt,
        adv_images=adv_images,
        adv_image_info=adv_image_info,
        adv_texts=adv_texts,
    )
    return attacked_samples, _attack_scope_debug(scope, need_img, need_txt, adv_images, adv_texts, adv_image_info, l2_values, linf_values, text_changed_flags)


# 执行 `样本 图像 key` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _sample_image_key(sample: Sample) -> str:
    return str(sample.metadata.get("source_image") or sample.metadata.get("image_id") or sample.sample_id)


# 确定 `图像 maps`，约束图像分支和文本分支的实际执行范围。
def _scope_image_maps(clean_samples: list[Sample]) -> tuple[dict[str, Sample], dict[str, np.ndarray]]:
    image_to_anchor: dict[str, Sample] = {}
    image_to_clean: dict[str, np.ndarray] = {}
    for sample in clean_samples:
        key = _sample_image_key(sample)
        if key not in image_to_anchor:
            image_to_anchor[key] = sample
            image_to_clean[key] = np.asarray(sample.image)
    return image_to_anchor, image_to_clean


# 执行 `record 作用范围 图像` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _record_scope_image(
    image_id: str,
    attacked: Any,
    image_to_clean: dict[str, np.ndarray],
    adv_images: dict[str, np.ndarray],
    adv_image_info: dict[str, dict[str, Any]],
    l2_values: list[float],
    linf_values: list[float],
) -> None:
    adv_images[image_id] = np.asarray(attacked.sample.image)
    try:
        adv_image_info[image_id] = {
            "patch_source": str(attacked.metadata.get("patch_source", "")),
            "registry_key": str(attacked.metadata.get("registry_key", "")),
            "patch_path": str(attacked.metadata.get("patch_path", "")),
            "implementation": str(attacked.metadata.get("implementation", "")),
        }
    except (AttributeError, TypeError):
        adv_image_info[image_id] = {}
    try:
        delta = np.asarray(attacked.sample.image, dtype=np.float32) - np.asarray(image_to_clean[image_id], dtype=np.float32)
        l2_values.append(float(np.linalg.norm(delta.reshape(-1), ord=2)))
        linf_values.append(float(np.max(np.abs(delta))) if delta.size else 0.0)
    except (KeyError, TypeError, ValueError):
        l2_values.append(float(attacked.perturbation_l2))
        linf_values.append(float(attacked.perturbation_linf))


# 推断 `攻击 unique 图像`，从样本、配置或运行记录中提取统一名称。
def _attack_unique_images(
    *,
    cfg: AppConfig,
    run_dir: str,
    attack_debug_root: Path,
    attack: Any,
    surrogate_adapter: Any,
    image_to_anchor: dict[str, Sample],
    image_to_clean: dict[str, np.ndarray],
    debug_enabled: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], list[float], list[float]]:
    adv_images: dict[str, np.ndarray] = {}
    adv_image_info: dict[str, dict[str, Any]] = {}
    l2_values: list[float] = []
    linf_values: list[float] = []
    ctx = AttackContext(config=cfg, model_adapter=surrogate_adapter, surrogate_model_adapter=surrogate_adapter, run_dir=run_dir, sample_debug_dir="")
    if bool(hasattr(attack, "attack_batch") and not debug_enabled):
        items = list(image_to_anchor.items())
        batch_size = max(1, int(getattr(cfg.attack, "batch_size", 16) or 16))
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            attacked_batch = attack.attack_batch([anchor for _, anchor in chunk], ctx)
            if len(attacked_batch) != len(chunk):
                raise RuntimeError("batch attack output length mismatch")
            for (image_id, _anchor), attacked in zip(chunk, attacked_batch):
                _record_scope_image(image_id, attacked, image_to_clean, adv_images, adv_image_info, l2_values, linf_values)
        return adv_images, adv_image_info, l2_values, linf_values
    for image_id, anchor in image_to_anchor.items():
        dbg = attack_debug_root / f"img_{_sanitize_dir_name(image_id)}"
        ctx.sample_debug_dir = str(dbg) if debug_enabled else ""
        attacked = attack.attack(anchor, ctx)
        _record_scope_image(image_id, attacked, image_to_clean, adv_images, adv_image_info, l2_values, linf_values)
    return adv_images, adv_image_info, l2_values, linf_values


# 推断 `攻击 作用范围 texts`，从样本、配置或运行记录中提取统一名称。
def _attack_scope_texts(
    *,
    cfg: AppConfig,
    run_dir: str,
    attack_debug_root: Path,
    clean_samples: list[Sample],
    attack: Any,
    surrogate_adapter: Any,
    debug_enabled: bool,
) -> tuple[dict[str, str], list[float]]:
    adv_texts: dict[str, str] = {}
    text_changed_flags: list[float] = []
    for sample in clean_samples:
        dbg = attack_debug_root / f"txt_{_sanitize_dir_name(sample.sample_id)}"
        attacked = attack.attack(
            sample,
            AttackContext(config=cfg, model_adapter=surrogate_adapter, surrogate_model_adapter=surrogate_adapter, run_dir=run_dir, sample_debug_dir=str(dbg) if debug_enabled else ""),
        )
        adv_text = str(attacked.sample.text)
        adv_texts[str(sample.sample_id)] = adv_text
        text_changed_flags.append(1.0 if adv_text != str(sample.text) else 0.0)
    return adv_texts, text_changed_flags


# 执行 `assemble 攻击后样本 作用范围 样本` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _assemble_attacked_scope_samples(
    *,
    cfg: AppConfig,
    clean_samples: list[Sample],
    scope: str,
    need_img: bool,
    need_txt: bool,
    adv_images: dict[str, np.ndarray],
    adv_image_info: dict[str, dict[str, Any]],
    adv_texts: dict[str, str],
) -> list[Sample]:
    attacked_samples: list[Sample] = []
    for sample in clean_samples:
        image_id = _sample_image_key(sample)
        meta = _scope_sample_metadata(cfg, sample, scope, need_img, image_id, adv_image_info)
        attacked_samples.append(
            Sample(
                sample_id=str(sample.sample_id),
                image=np.asarray(adv_images.get(image_id, sample.image) if need_img else sample.image),
                text=str(adv_texts.get(str(sample.sample_id), sample.text) if need_txt else sample.text),
                target_text=str(sample.target_text or ""),
                metadata=meta,
            )
        )
    return attacked_samples


# 确定 `样本 metadata`，约束图像分支和文本分支的实际执行范围。
def _scope_sample_metadata(cfg: AppConfig, sample: Sample, scope: str, need_img: bool, image_id: str, adv_image_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = dict(sample.metadata)
    meta.update({"attack_scope": scope, "attack_name": str(getattr(cfg.plugins, "attack", "attack")), "attack_mode": str(getattr(cfg.attack, "mode", "A"))})
    if need_img and image_id in adv_image_info:
        info = adv_image_info.get(image_id, {}) or {}
        if info.get("patch_source"):
            meta["patch_source"] = str(info.get("patch_source"))
        if info.get("registry_key"):
            meta["registry_key"] = str(info.get("registry_key"))
    return meta


# 推断 `攻击 作用范围 调试`，从样本、配置或运行记录中提取统一名称。
def _attack_scope_debug(
    scope: str,
    need_img: bool,
    need_txt: bool,
    adv_images: dict[str, np.ndarray],
    adv_texts: dict[str, str],
    adv_image_info: dict[str, dict[str, Any]],
    l2_values: list[float],
    linf_values: list[float],
    text_changed_flags: list[float],
) -> dict[str, Any]:
    first_info = next(iter(adv_image_info.values()), {}) if adv_image_info else {}
    return {
        "scope": scope,
        "need_image": need_img,
        "need_text": need_txt,
        "num_images_attacked": int(len(adv_images)),
        "num_texts_attacked": int(len(adv_texts)),
        "avg_l2": float(_safe_mean(l2_values)),
        "avg_linf": float(_safe_mean(linf_values)),
        "text_changed_ratio": float(_safe_mean(text_changed_flags)),
        "l2_values": l2_values,
        "linf_values": linf_values,
        "patch_source": str(first_info.get("patch_source", "")),
        "registry_key": str(first_info.get("registry_key", "")),
    }


# 执行 `defend 样本` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _defend_samples(
    *,
    cfg: AppConfig,
    run_dir: str,
    samples: list[Sample],
    defense: Any,
    adapter: Any,
    stage: str,
) -> tuple[list[Sample], dict[str, Any]]:
    defended_samples: list[Sample] = []
    text_deltas: list[float] = []
    debug_root = Path(run_dir) / "attack_debug"
    debug_root.mkdir(parents=True, exist_ok=True)
    debug_enabled = _debug_artifacts_enabled(cfg)

    for s in samples:
        dbg = debug_root / f"def_{_sanitize_dir_name(stage)}_{_sanitize_dir_name(s.sample_id)}"
        defended = defense.defend(
            s,
            DefenseContext(
                config=cfg,
                model_adapter=adapter,
                stage=stage,
                run_dir=run_dir,
                sample_debug_dir=str(dbg) if debug_enabled else "",
            ),
        )
        ds = defended.sample
        meta = dict(ds.metadata)
        meta["defense_name"] = str(cfg.plugins.defense)
        meta["defense_stage"] = str(stage)
        defended_samples.append(
            Sample(
                sample_id=str(ds.sample_id),
                image=np.asarray(ds.image, dtype=np.float32),
                text=str(ds.text),
                target_text=str(ds.target_text or ""),
                metadata=meta,
            )
        )
        text_deltas.append(float(0.0 if str(s.text) == str(ds.text) else 1.0))

    return defended_samples, {
        "stage": str(stage),
        "num_samples": int(len(defended_samples)),
        "text_changed_ratio": float(_safe_mean(text_deltas)),
    }


# 标记 `error values` 阶段，区分 clean、attacked 和 defended 样本。
def _stage_error_values(victim_metrics: dict[str, Any], victim_names: list[str], stage: str) -> list[float]:
    values: list[float] = []
    for victim_name in victim_names:
        metrics = victim_metrics.get(victim_name, {}).get(stage, {})
        if isinstance(metrics, dict):
            values.append(float(metrics.get("ir_asr@1", 0.0)))
            values.append(float(metrics.get("tr_asr@1", 0.0)))
    return values


# 计算 `ASR values` 条件指标，只在满足前置条件的样本上统计。
def _conditional_asr_values(victim_metrics: dict[str, Any], victim_names: list[str]) -> list[float]:
    values: list[float] = []
    for victim_name in victim_names:
        conditional = victim_metrics.get(victim_name, {}).get("conditional", {})
        if isinstance(conditional, dict):
            values.append(float(conditional.get("ir_cond_asr@1", 0.0)))
            values.append(float(conditional.get("tr_cond_asr@1", 0.0)))
    return values


# 计算 `delta 摘要` 排名，使用稳定排序保证指标结果可复现。
def _rank_delta_summary(victim_metrics: dict[str, Any], victim_names: list[str]) -> tuple[list[float], float]:
    rank_deltas: list[float] = []
    worst_victim_asr = 0.0
    for victim_name in victim_names:
        clean = victim_metrics.get(victim_name, {}).get("clean", {}) or {}
        attacked = victim_metrics.get(victim_name, {}).get("attacked", {}) or {}
        conditional = victim_metrics.get(victim_name, {}).get("conditional", {}) or {}
        victim_attack_success = _victim_conditional_asr_at_1(conditional) if isinstance(conditional, dict) and conditional else _victim_stage_asr_at_1(attacked)
        worst_victim_asr = max(float(worst_victim_asr), float(victim_attack_success))
        rank_deltas.append(
            0.5
            * (
                float(attacked.get("mean_rank_ir", 0.0)) - float(clean.get("mean_rank_ir", 0.0))
                + float(attacked.get("mean_rank_tr", 0.0)) - float(clean.get("mean_rank_tr", 0.0))
            )
        )
    return rank_deltas, float(worst_victim_asr)


# 推断 `summarize 攻击 outcomes`，从样本、配置或运行记录中提取统一名称。
def _summarize_attack_outcomes(
    *,
    victim_metrics: dict[str, Any],
    victim_names: list[str],
    eval_scope: str,
    attack_debug: dict[str, Any],
    transfer_threshold: float,
) -> dict[str, Any]:
    l2_values = list(attack_debug.get("l2_values", [])) if isinstance(attack_debug, dict) else []
    linf_values = list(attack_debug.get("linf_values", [])) if isinstance(attack_debug, dict) else []

    attacked_error_values = _stage_error_values(victim_metrics, victim_names, "attacked") if eval_scope != "clean" else []
    defended_error_values = _stage_error_values(victim_metrics, victim_names, "defended_attack") if eval_scope != "clean" else []

    attacked_error_rate_at1 = float(_safe_mean(attacked_error_values))
    defended_error_rate_at1 = (
        float(_safe_mean(defended_error_values)) if defended_error_values else attacked_error_rate_at1
    )

    conditional_asr_values = _conditional_asr_values(victim_metrics, victim_names) if eval_scope != "clean" else []
    conditional_asr_attack = float(_safe_mean(conditional_asr_values))

    asr_attack = conditional_asr_attack
    asr_defended = conditional_asr_attack if not defended_error_values else defended_error_rate_at1
    defense_gain = float(asr_attack - asr_defended)
    transfer_score, victim_transfer_asrs = _victim_transfer_score(
        victim_metrics,
        victim_names,
        threshold=float(transfer_threshold),
    )

    rank_deltas, worst_victim_asr = _rank_delta_summary(victim_metrics, victim_names)

    return {
        "l2_values": l2_values,
        "linf_values": linf_values,
        "attacked_error_rate_at1": attacked_error_rate_at1,
        "defended_error_rate_at1": defended_error_rate_at1,
        "conditional_asr_attack": conditional_asr_attack,
        "asr_attack": asr_attack,
        "asr_defended": asr_defended,
        "defense_gain": defense_gain,
        "transfer_score": float(transfer_score),
        "victim_transfer_asrs": victim_transfer_asrs,
        "rank_deltas": rank_deltas,
        "rank_delta": float(_safe_mean(rank_deltas)),
        "worst_victim_asr": float(worst_victim_asr),
        "avg_l2_all": float(_safe_mean(l2_values)),
        "avg_linf_all": float(_safe_mean(linf_values)),
        "text_changed_ratio": float(attack_debug.get("text_changed_ratio", 0.0) or 0.0)
        if isinstance(attack_debug, dict)
        else 0.0,
    }


# 加载 `图文检索 样本 and 索引`，把外部文件、配置或运行产物转换为内存结构。
def _load_vlr_samples_and_index(cfg: AppConfig, progress: Callable[[str, str, float | None, str], None] | None) -> tuple[list[Sample], VLRIndex]:
    _emit_progress(progress, "dataset_loading", "running", 32, "正在装载数据集并构建检索索引。")
    dataset = load_dataset(cfg)
    if cfg.runner.max_samples > 0:
        dataset = dataset[: cfg.runner.max_samples]
    LOG.info("Loaded dataset: %d samples", len(dataset))

    clean_samples = list(dataset)
    clean_index = build_vlr_index(clean_samples)
    clean_index = _index_downsample(clean_index, int(cfg.runner.max_pairs), seed=cfg.seed)
    keep_text_ids = set(clean_index.text_ids)
    clean_samples_subset = [s for s in clean_samples if str(s.sample_id) in keep_text_ids]
    clean_index = build_vlr_index(clean_samples_subset)
    _emit_progress(progress, "dataset_loading", "success", 38, f"数据集装载完成，共纳入 {len(clean_samples_subset)} 条样本。")
    return clean_samples_subset, clean_index


# 评估 `clean 图文检索 stage` 结果，汇总攻击前后指标和风险证据。
def _evaluate_clean_vlr_stage(
    *,
    victims: dict[str, Any],
    victim_names: list[str],
    clean_index: VLRIndex,
    ks: list[int],
    batch_size: int,
    progress: Callable[[str, str, float | None, str], None] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], dict[str, dict[str, str]]]:
    _emit_progress(progress, "victim_evaluation", "running", 46, "正在评测正常输入在各受测模型上的表现。")
    clean_sims, clean_metrics = _evaluate_victim_stage(
        victims=victims,
        victim_names=victim_names,
        index=clean_index,
        ks=ks,
        batch_size=batch_size,
        progress=progress,
        progress_start=46,
        progress_end=56,
        progress_message="正在评测正常输入在各受测模型上的表现",
    )
    victim_metrics: dict[str, Any] = {}
    results_rows: list[dict[str, Any]] = []
    victim_status: dict[str, dict[str, str]] = {}
    for victim_name in victim_names:
        metrics = clean_metrics[victim_name]
        victim_status.setdefault(victim_name, {})["clean"] = "ok"
        victim_metrics.setdefault(victim_name, {})["clean"] = metrics
        results_rows.append({"victim": victim_name, "scope": "clean", **metrics})
    return clean_sims, victim_metrics, results_rows, victim_status


# 执行 `staged lifecycle enabled` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _staged_lifecycle_enabled(cfg: AppConfig) -> bool:
    return bool(getattr(cfg.runner, "staged_model_lifecycle", True))


# 推断 `release 本地 视觉语言模型 所属 图文检索 攻击`，从样本、配置或运行记录中提取统一名称。
def _release_local_vlm_for_vlr_attack(cfg: AppConfig, progress: Callable[[str, str, float | None, str], None] | None) -> None:
    if not _staged_lifecycle_enabled(cfg) or not bool(getattr(cfg.runner, "stop_local_vlm_before_attack", True)):
        return
    _emit_progress(progress, "model_preflight", "running", 57, "正在停止本地 VLM，释放显存给攻击生成阶段。")
    stop_local_vlm_servers()
    empty_cuda_cache()


# 准备 `图文检索 本地 victims` 数据，补齐后续运行、报告或测试需要的字段。
def _prepare_vlr_local_victims(cfg: AppConfig, victim_names: list[str], progress: Callable[[str, str, float | None, str], None] | None) -> None:
    if not _staged_lifecycle_enabled(cfg):
        return
    empty_cuda_cache()
    local_victims = local_vlm_adapters(victim_names)
    if not local_victims or not bool(getattr(cfg.runner, "restart_local_vlm_for_evaluation", True)):
        return
    _emit_progress(progress, "model_preflight", "running", 64, "攻击图已生成，正在启动本地 VLM 评测攻击样本。")
    ensure_local_vlm_adapters_ready(local_victims)
    _emit_progress(progress, "model_preflight", "success", 66, "本地 VLM 已就绪，继续检索评测。")


# 执行 `攻击后样本 图文检索 stage` 流程，按配置驱动评测运行器完成一次任务。
def _run_attacked_vlr_stage(cfg: AppConfig, ctx: dict[str, Any], progress: Callable[[str, str, float | None, str], None] | None) -> dict[str, Any]:
    eval_scope = str(ctx["eval_scope"])
    state: dict[str, Any] = {
        "attack_debug": {},
        "attacked_index": None,
        "attacked_sims": {},
        "attacked_samples_subset": [],
        "defended_attack_debug": {},
        "defended_attack_index": None,
        "defended_attack_samples": [],
    }
    if eval_scope == "clean":
        return state

    _release_local_vlm_for_vlr_attack(cfg, progress)
    _emit_progress(progress, "attack_execution", "running", 58, "正在生成对抗样本。")
    attacked_samples_subset, attack_debug = _attack_scope_samples(
        cfg=cfg,
        run_dir=ctx["run_dir"],
        clean_samples=ctx["clean_samples_subset"],
        clean_index=ctx["clean_index"],
        attack=ctx["attack"],
        surrogate_adapter=ctx["surrogate_adapter"],
        scope=eval_scope,
    )
    attacked_index = build_vlr_index(attacked_samples_subset)
    _prepare_vlr_local_victims(cfg, list(ctx["victim_names"]), progress)
    attacked_sims = _evaluate_attacked_victims(
        victims=ctx["victims"],
        victim_names=ctx["victim_names"],
        attacked_index=attacked_index,
        clean_index=ctx["clean_index"],
        clean_sims=ctx["clean_sims"],
        ks=ctx["ks"],
        batch_size=ctx["batch_size"],
        progress=progress,
        victim_metrics=ctx["victim_metrics"],
        results_rows=ctx["results_rows"],
        victim_status=ctx["victim_status"],
        eval_scope=eval_scope,
    )
    state.update(
        {
            "attack_debug": attack_debug,
            "attacked_index": attacked_index,
            "attacked_sims": attacked_sims,
            "attacked_samples_subset": attacked_samples_subset,
        }
    )
    if ctx["defense"] is not None and bool(cfg.defense.apply_on_attacked):
        defended_samples, defended_debug = _run_defended_attack_stage(
            cfg=cfg,
            run_dir=ctx["run_dir"],
            samples=attacked_samples_subset,
            victims=ctx["victims"],
            victim_names=ctx["victim_names"],
            surrogate_adapter=ctx["surrogate_adapter"],
            defense=ctx["defense"],
            ks=ctx["ks"],
            batch_size=ctx["batch_size"],
            progress=progress,
            victim_metrics=ctx["victim_metrics"],
            results_rows=ctx["results_rows"],
            victim_status=ctx["victim_status"],
        )
        state["defended_attack_debug"] = defended_debug
        state["defended_attack_index"] = build_vlr_index(defended_samples)
        state["defended_attack_samples"] = defended_samples
    return state


# 评估 `攻击后样本 victims` 结果，汇总攻击前后指标和风险证据。
def _evaluate_attacked_victims(
    *,
    victims: dict[str, Any],
    victim_names: list[str],
    attacked_index: VLRIndex,
    clean_index: VLRIndex,
    clean_sims: dict[str, np.ndarray],
    ks: list[int],
    batch_size: int,
    progress: Callable[[str, str, float | None, str], None] | None,
    victim_metrics: dict[str, Any],
    results_rows: list[dict[str, Any]],
    victim_status: dict[str, dict[str, str]],
    eval_scope: str,
) -> dict[str, np.ndarray]:
    _emit_progress(progress, "victim_evaluation", "running", 72, "正在评测攻击后输入在各受测模型上的表现。")
    attacked_sims, attacked_metrics = _evaluate_victim_stage(
        victims=victims,
        victim_names=victim_names,
        index=attacked_index,
        ks=ks,
        batch_size=batch_size,
        progress=progress,
        progress_start=72,
        progress_end=80,
        progress_message="正在评测攻击后输入在各受测模型上的表现",
    )
    for victim_name in victim_names:
        metrics = attacked_metrics[victim_name]
        victim_status.setdefault(victim_name, {})["attacked"] = "ok"
        victim_metrics.setdefault(victim_name, {})["attacked"] = metrics
        results_rows.append({"victim": victim_name, "scope": eval_scope, **metrics})
        conditional = conditional_attack_metrics(clean_sims[victim_name], attacked_sims[victim_name], clean_index, ks=ks)
        victim_metrics.setdefault(victim_name, {})["conditional"] = conditional
        results_rows.append({"victim": victim_name, "scope": "conditional", **conditional})
    return attacked_sims


# 执行 `defended 攻击 stage` 流程，按配置驱动评测运行器完成一次任务。
def _run_defended_attack_stage(
    *,
    cfg: AppConfig,
    run_dir: str,
    samples: list[Sample],
    victims: dict[str, Any],
    victim_names: list[str],
    surrogate_adapter: Any,
    defense: Any,
    ks: list[int],
    batch_size: int,
    progress: Callable[[str, str, float | None, str], None] | None,
    victim_metrics: dict[str, Any],
    results_rows: list[dict[str, Any]],
    victim_status: dict[str, dict[str, str]],
) -> tuple[list[Sample], dict[str, Any]]:
    _emit_progress(progress, "victim_evaluation", "running", 82, "正在评测防御后输入在各受测模型上的表现。")
    defended_samples, defended_debug = _defend_samples(
        cfg=cfg,
        run_dir=run_dir,
        samples=samples,
        defense=defense,
        adapter=surrogate_adapter,
        stage="defended_attack",
    )
    defended_index = build_vlr_index(defended_samples)
    _, defended_metrics = _evaluate_victim_stage(
        victims=victims,
        victim_names=victim_names,
        index=defended_index,
        ks=ks,
        batch_size=batch_size,
        progress=progress,
        progress_start=82,
        progress_end=84,
        progress_message="正在评测防御后输入在各受测模型上的表现",
    )
    for victim_name in victim_names:
        metrics = defended_metrics[victim_name]
        victim_status.setdefault(victim_name, {})["defended_attack"] = "ok"
        victim_metrics.setdefault(victim_name, {})["defended_attack"] = metrics
        results_rows.append({"victim": victim_name, "scope": "defended_attack", **metrics})
    return defended_samples, defended_debug


# 执行 `clean 防御 图文检索 stage` 流程，按配置驱动评测运行器完成一次任务。
def _run_clean_defense_vlr_stage(
    *,
    cfg: AppConfig,
    run_dir: str,
    clean_samples_subset: list[Sample],
    victims: dict[str, Any],
    victim_names: list[str],
    surrogate_adapter: Any,
    defense: Any | None,
    ks: list[int],
    batch_size: int,
    progress: Callable[[str, str, float | None, str], None] | None,
    victim_metrics: dict[str, Any],
    results_rows: list[dict[str, Any]],
    victim_status: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], VLRIndex | None]:
    if defense is None or not bool(cfg.defense.apply_on_clean):
        return {}, None
    _emit_progress(progress, "victim_evaluation", "running", 84, "正在统计防御对正常输入的影响。")
    defended_samples, defended_debug = _defend_samples(
        cfg=cfg,
        run_dir=run_dir,
        samples=clean_samples_subset,
        defense=defense,
        adapter=surrogate_adapter,
        stage="defended_clean",
    )
    defended_index = build_vlr_index(defended_samples)
    _, defended_metrics = _evaluate_victim_stage(
        victims=victims,
        victim_names=victim_names,
        index=defended_index,
        ks=ks,
        batch_size=batch_size,
        progress=progress,
        progress_start=84,
        progress_end=88,
        progress_message="正在统计防御对正常输入的影响",
    )
    for victim_name in victim_names:
        metrics = defended_metrics[victim_name]
        victim_status.setdefault(victim_name, {})["defended_clean"] = "ok"
        victim_metrics.setdefault(victim_name, {})["defended_clean"] = metrics
        results_rows.append({"victim": victim_name, "scope": "defended_clean", **metrics})
    return defended_debug, defended_index


# 构建 `图文检索 failure rows` 数据，集中整理评测运行器需要的输出结构。
def _build_vlr_failure_rows(*, eval_scope: str, attacked_index: VLRIndex | None, attacked_sims: dict[str, np.ndarray], ks: list[int]) -> list[dict[str, Any]]:
    if eval_scope == "clean" or attacked_index is None:
        return []
    topk = max(5, max(int(x) for x in ks if int(x) > 0))
    failure_rows: list[dict[str, Any]] = []
    for victim_name, sim in attacked_sims.items():
        top = topk_indices_t2i(sim, k=topk)
        gt = attacked_index.gt_img_idx
        for ti in range(len(attacked_index.texts)):
            hit = bool(int(gt[ti]) in [int(x) for x in top[ti, :5].tolist()])
            failure_rows.append(
                {
                    "victim": victim_name,
                    "scope": eval_scope,
                    "query_type": "t2i",
                    "text_id": attacked_index.text_ids[ti],
                    "text": attacked_index.texts[ti],
                    "gt_image_id": attacked_index.image_ids[int(gt[ti])],
                    "top5_image_ids": [attacked_index.image_ids[int(x)] for x in top[ti, :5].tolist()],
                    "judge_success": hit,
                    "judge_reason": "gt_in_top5" if hit else "gt_not_in_top5",
                }
            )
    return failure_rows


# 推断 `图文检索 conditional ASR 攻击`，从样本、配置或运行记录中提取统一名称。
def _vlr_conditional_asr_attack(victim_metrics: dict[str, Any], victim_names: list[str], eval_scope: str) -> float:
    values: list[float] = []
    if eval_scope == "clean":
        return 0.0
    for victim_name in victim_names:
        cond_m = victim_metrics.get(victim_name, {}).get("conditional", {})
        if isinstance(cond_m, dict):
            values.append(float(cond_m.get("ir_cond_asr@1", 0.0)))
            values.append(float(cond_m.get("tr_cond_asr@1", 0.0)))
    return float(_safe_mean(values))


# 组装 `图文检索 风险 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _vlr_risk_payload(cfg: AppConfig, *, asr_attack: float, semantic_score: float, cost_score: float, transfer_score: float, stability_score: float) -> dict[str, Any]:
    if bool(cfg.risk.enabled):
        return compute_risk_score(
            scenario=str(cfg.risk.scenario or "retrieval"),
            components={
                "effectiveness": float(asr_attack),
                "semantic": float(semantic_score),
                "cost": float(cost_score),
                "transfer": float(transfer_score),
                "stability": float(stability_score),
            },
            weights=dict(cfg.risk.weights or {}),
        )
    return {
        "risk_score": 0.0,
        "risk_level": "disabled",
        "risk_scenario": str(cfg.risk.scenario or "retrieval"),
        "risk_breakdown": {},
        "risk_weights": {},
        "risk_recommendations": [],
    }


# 整理 `图文检索 outcome context`，描述当前服务器运行环境、模型入口或部署状态。
def _vlr_outcome_context(
    *,
    cfg: AppConfig,
    surrogate_adapter: Any,
    clean_index: VLRIndex,
    attacked_index: VLRIndex | None,
    clean_samples_subset: list[Sample],
    attacked_samples_subset: list[Sample],
    victim_metrics: dict[str, Any],
    victim_names: list[str],
    eval_scope: str,
    attack_debug: dict[str, Any],
) -> dict[str, Any]:
    transfer_threshold = float(getattr(cfg.risk, "transfer_success_threshold", 0.2) or 0.2)
    outcome = _summarize_attack_outcomes(
        victim_metrics=victim_metrics,
        victim_names=victim_names,
        eval_scope=eval_scope,
        attack_debug=attack_debug,
        transfer_threshold=transfer_threshold,
    )
    avg_l2_all = float(outcome["avg_l2_all"])
    avg_linf_all = float(outcome["avg_linf_all"])
    semantic_preservation = _semantic_preservation(
        surrogate_adapter=surrogate_adapter,
        clean_index=clean_index,
        attacked_index=attacked_index,
        avg_linf=avg_linf_all,
        cfg=cfg,
    ) if eval_scope != "clean" else {"available": False, "reason": "clean-only run"}
    object_decision_proxy = _object_decision_proxy(
        adapter=surrogate_adapter,
        clean_samples=clean_samples_subset,
        attacked_samples=attacked_samples_subset,
    ) if eval_scope != "clean" and attacked_samples_subset else {"available": False, "reason": "clean-only run"}
    text_changed_ratio = float(outcome["text_changed_ratio"])
    semantic_component = float(
        semantic_preservation.get(
            "combined_semantic_preservation",
            0.5 * ((1.0 - max(0.0, min(1.0, text_changed_ratio))) + normalize_inverse(avg_linf_all, float(cfg.risk.linf_reference))),
        )
        or 0.0
    )
    rank_deltas = list(outcome["rank_deltas"])
    context = dict(outcome)
    context.update(
        {
            "transfer_threshold": transfer_threshold,
            "semantic_preservation": semantic_preservation,
            "object_decision_proxy": object_decision_proxy,
            "semantic_score": max(0.0, min(1.0, semantic_component)),
            "cost_score": 0.5 * (normalize_inverse(avg_l2_all, float(cfg.risk.l2_reference)) + normalize_inverse(avg_linf_all, float(cfg.risk.linf_reference))),
            "stability_score": max(normalize_direct(float(_safe_mean(rank_deltas)), float(cfg.risk.rank_reference)), float(outcome["worst_victim_asr"])),
            "conditional_asr_attack": _vlr_conditional_asr_attack(victim_metrics, victim_names, eval_scope),
        }
    )
    context["risk_payload"] = _vlr_risk_payload(
        cfg,
        asr_attack=float(context["asr_attack"]),
        semantic_score=float(context["semantic_score"]),
        cost_score=float(context["cost_score"]),
        transfer_score=float(context["transfer_score"]),
        stability_score=float(context["stability_score"]),
    )
    return context


# 整理 `victim 对比 rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _victim_compare_rows(victim_metrics: dict[str, Any], victim_names: list[str], victim_status: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for victim_name in victim_names:
        clean_m = victim_metrics.get(victim_name, {}).get("clean", {}) or {}
        adv_m = victim_metrics.get(victim_name, {}).get("attacked", {}) or {}
        rows.append(
            {
                "victim": victim_name,
                "status": victim_status.get(victim_name, {}),
                "clean": clean_m,
                "attacked": adv_m,
                "conditional": victim_metrics.get(victim_name, {}).get("conditional", {}) or {},
                "delta_mean_rank_ir": float(adv_m.get("mean_rank_ir", 0.0)) - float(clean_m.get("mean_rank_ir", 0.0)),
                "delta_mean_rank_tr": float(adv_m.get("mean_rank_tr", 0.0)) - float(clean_m.get("mean_rank_tr", 0.0)),
            }
        )
    return rows


# 整理 `防御 对比 rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _defense_compare_rows(victim_metrics: dict[str, Any], victim_names: list[str], ks: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for victim_name in victim_names:
        clean_m = victim_metrics.get(victim_name, {}).get("clean", {}) or {}
        adv_m = victim_metrics.get(victim_name, {}).get("attacked", {}) or {}
        def_a = victim_metrics.get(victim_name, {}).get("defended_attack", {}) or {}
        def_c = victim_metrics.get(victim_name, {}).get("defended_clean", {}) or {}
        rec_map: dict[str, float] = {}
        util_map: dict[str, float] = {}
        for k in ks:
            ck = float(clean_m.get(f"ir_r@{k}", 0.0))
            ak = float(adv_m.get(f"ir_r@{k}", 0.0))
            dk = float(def_a.get(f"ir_r@{k}", 0.0))
            dck = float(def_c.get(f"ir_r@{k}", 0.0))
            rec_map[f"defense_recovery@{k}"] = float(dk - ak)
            util_map[f"defense_utility_drop@{k}"] = float(ck - dck)
        rows.append(
            {
                "victim": victim_name,
                "attack_drop_r1": float(clean_m.get("ir_r@1", 0.0)) - float(adv_m.get("ir_r@1", 0.0)),
                "defense_recovery_r1": float(def_a.get("ir_r@1", 0.0)) - float(adv_m.get("ir_r@1", 0.0)),
                **rec_map,
                **util_map,
            }
        )
    return rows


# 整理 `reproduction fidelity rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _reproduction_fidelity_rows() -> list[dict[str, str]]:
    return [
        {"paper": "AdvCLIP", "status": "approx", "source": "src/mmsec_eval/attacks/advclip/*"},
        {"paper": "TMM", "status": "approx", "source": "src/mmsec_eval/attacks/tmm/*"},
        {"paper": "ADVEDM", "status": "approx", "source": "src/mmsec_eval/attacks/advedm/*"},
    ]


# 构建 `图文检索 摘要 载荷` 数据，集中整理评测运行器需要的输出结构。
def _build_vlr_summary_payload(
    *,
    cfg: AppConfig,
    run_id: str,
    clean_index: VLRIndex,
    surrogate_name: str,
    victim_names: list[str],
    victim_metrics: dict[str, Any],
    victim_status: dict[str, dict[str, str]],
    victim_failures: list[dict[str, str]],
    victim_compare: list[dict[str, Any]],
    defense_compare_rows: list[dict[str, Any]],
    feature_projection: dict[str, Any],
    attack_debug: dict[str, Any],
    defended_attack_debug: dict[str, Any],
    defended_clean_debug: dict[str, Any],
    metric_quality: dict[str, Any],
    outcome: dict[str, Any],
    defense: Any | None,
    ks: list[int],
    eval_scope: str,
) -> dict[str, Any]:
    risk_payload = outcome["risk_payload"]
    reproduction_rows = _reproduction_fidelity_rows()
    return {
        "run_id": run_id,
        "task_kind": "vlr",
        "retrieval_k": ks,
        "compare_stages": list(cfg.task.compare_stages or ["clean", "attacked", "defended"]),
        "eval_scope": eval_scope,
        "num_images": len(clean_index.images),
        "num_texts": len(clean_index.texts),
        "attack": str(cfg.plugins.attack),
        "attack_mode": str(cfg.attack.mode),
        "defense": str(cfg.plugins.defense) if defense is not None else "",
        "defense_enabled": bool(defense is not None),
        "experiment_id": str(cfg.runner.experiment_id or ""),
        "model_adapter": str(cfg.plugins.model_adapter),
        "surrogate_model_adapter": surrogate_name,
        "victim_model_adapters": victim_names,
        "asr": round(float(outcome["asr_attack"]), 6),
        "asr_attack": round(float(outcome["asr_attack"]), 6),
        "conditional_asr_attack": round(float(outcome["conditional_asr_attack"]), 6),
        "asr_definition": "conditional_clean_top1_drop",
        "attacked_error_rate@1": round(float(outcome["attacked_error_rate_at1"]), 6),
        "defended_error_rate@1": round(float(outcome["defended_error_rate_at1"]), 6),
        "unconditional_asr_attack": round(float(outcome["attacked_error_rate_at1"]), 6),
        "asr_defended": round(float(outcome["asr_defended"]), 6),
        "defense_gain": round(float(outcome["defense_gain"]), 6),
        "avg_l2": round(float(_safe_mean(list(outcome["l2_values"]))), 6),
        "avg_linf": round(float(_safe_mean(list(outcome["linf_values"]))), 6),
        "transfer_success_rate": round(float(outcome["transfer_score"]), 6),
        "semantic_preservation": outcome["semantic_preservation"],
        "object_decision_proxy": outcome["object_decision_proxy"],
        "metric_quality": metric_quality,
        **risk_payload,
        "dataset_name": str(cfg.dataset.kind),
        "benchmark_tag": str(cfg.dataset.benchmark_tag or cfg.dataset.kind),
        "victims": victim_metrics,
        "victim_status": victim_status,
        "num_victim_failures": int(len(victim_failures)),
        "victim_failures": victim_failures[:20],
        "victim_compare": victim_compare,
        "defense_compare": defense_compare_rows,
        "risk": _vlr_summary_risk_payload(outcome),
        "feature_projection": feature_projection,
        "attack_debug": attack_debug,
        "defense_debug": {"defended_attack": defended_attack_debug, "defended_clean": defended_clean_debug},
        "reproduction_fidelity": {row["paper"].lower(): row["status"] for row in reproduction_rows},
    }


# 组装 `图文检索 摘要 风险 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _vlr_summary_risk_payload(outcome: dict[str, Any]) -> dict[str, Any]:
    object_decision_proxy = outcome["object_decision_proxy"]
    return {
        **outcome["risk_payload"],
        "components_raw": {
            "avg_l2": float(outcome["avg_l2_all"]),
            "avg_linf": float(outcome["avg_linf_all"]),
            "text_changed_ratio": float(outcome["text_changed_ratio"]),
            "semantic_preservation": float(outcome["semantic_score"]),
            "conditional_asr_attack@1": float(outcome["conditional_asr_attack"]),
            "attacked_error_rate@1": float(outcome["attacked_error_rate_at1"]),
            "defended_error_rate@1": float(outcome["defended_error_rate_at1"]),
            "object_decision_valid_wrong_rate": float(object_decision_proxy.get("valid_wrong_rate", 0.0) or 0.0) if isinstance(object_decision_proxy, dict) else 0.0,
            "avg_rank_delta": float(_safe_mean(list(outcome["rank_deltas"]))),
            "worst_victim_asr@1": float(outcome["worst_victim_asr"]),
            "transfer_success_threshold": float(outcome["transfer_threshold"]),
            "victim_transfer_asr@1": [float(x) for x in list(outcome["victim_transfer_asrs"])],
        },
    }


# 构建 `图文检索 报告 载荷` 数据，集中整理评测运行器需要的输出结构。
def _build_vlr_report_payload(
    *,
    summary: dict[str, Any],
    cfg: AppConfig,
    clean_index: VLRIndex,
    victim_names: list[str],
    victim_metrics: dict[str, Any],
    victim_status: dict[str, dict[str, str]],
    victim_failures: list[dict[str, str]],
    victim_compare: list[dict[str, Any]],
    defense_compare_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    feature_projection: dict[str, Any],
    metric_quality: dict[str, Any],
    outcome: dict[str, Any],
) -> dict[str, Any]:
    mode_key = f"{cfg.plugins.attack}:{cfg.attack.mode}"
    return {
        "summary": summary,
        "mode_stats": {mode_key: {"count": float(len(clean_index.texts)), "asr": float(outcome["asr_attack"]), "attacked_error_rate@1": float(outcome["attacked_error_rate_at1"])}},
        "stage_metrics": _vlr_stage_metrics(victim_metrics, victim_names),
        "defense_compare": defense_compare_rows,
        "risk": summary.get("risk", {}),
        "semantic_preservation": outcome["semantic_preservation"],
        "object_decision_proxy": outcome["object_decision_proxy"],
        "metric_quality": metric_quality,
        "feature_projection": feature_projection,
        "rows_preview": failure_rows,
        "metric_series": {"l2": list(outcome["l2_values"]), "linf": list(outcome["linf_values"])},
        "vlr": {
            **_vlr_stage_metrics(victim_metrics, victim_names),
            "victim_status": victim_status,
            "victim_failures": victim_failures,
            "victim_compare": victim_compare,
            "failure_cases": failure_rows,
        },
        "reproduction_fidelity": _reproduction_fidelity_rows(),
    }


# 执行 `as 案例 分数 文本` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _as_case_score_text(score: float | None) -> str:
    if score is None:
        return "未记录模型分数"
    return f"CLIP 相似度（CLIP similarity）={float(score):.4f}"


# 计算 `样本 delta 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _sample_delta_metrics(clean: Sample, attacked: Sample) -> tuple[int, float, float]:
    try:
        delta = np.asarray(attacked.image, dtype=np.float32) - np.asarray(clean.image, dtype=np.float32)
        flat = np.asarray(delta).reshape(-1)
        l0 = int(np.count_nonzero(np.abs(flat) > 1e-8))
        l2 = float(np.linalg.norm(flat, ord=2))
        linf = float(np.max(np.abs(flat))) if flat.size else 0.0
        return l0, l2, linf
    except (TypeError, ValueError):
        return 0, 0.0, 0.0


# 整理 `证据包 rows` 字段，统一生成式案例在 runner 内的读取口径。
def _case_bundle_rows(
    *,
    adapter: Any,
    clean_samples: list[Sample],
    attacked_samples: list[Sample],
    defended_samples: list[Sample],
    max_cases: int,
) -> list[dict[str, Any]]:
    attacked_by_id = {str(sample.sample_id): sample for sample in attacked_samples}
    defended_by_id = {str(sample.sample_id): sample for sample in defended_samples}
    rows: list[dict[str, Any]] = []
    for clean in clean_samples:
        sid = str(clean.sample_id)
        attacked = attacked_by_id.get(sid)
        if attacked is None:
            continue
        defended = defended_by_id.get(sid)
        clean_score = _score_image_text(adapter, np.asarray(clean.image, dtype=np.float32), str(clean.text))
        adv_score = _score_image_text(adapter, np.asarray(attacked.image, dtype=np.float32), str(attacked.text))
        defended_score = (
            _score_image_text(adapter, np.asarray(defended.image, dtype=np.float32), str(defended.text))
            if defended is not None
            else None
        )
        drop = float((clean_score or 0.0) - (adv_score or 0.0))
        rows.append(
            {
                "sample_id": sid,
                "clean": clean,
                "attacked": attacked,
                "defended": defended,
                "clean_score": clean_score,
                "adv_score": adv_score,
                "defended_score": defended_score,
                "score_drop": drop,
            }
        )
    rows.sort(key=lambda item: float(item["score_drop"]), reverse=True)
    return rows if int(max_cases) <= 0 else rows[: max(0, int(max_cases))]


# 写出 `图文检索 案例 图像`，保证后续报告、页面或复现实验能读取。
def _save_vlr_case_images(
    case_dir: Path,
    clean: Sample,
    attacked: Sample,
    defended: Sample | None,
) -> dict[str, str]:
    refs = {
        "clean_image": save_image_png(str(case_dir / "clean.png"), np.asarray(clean.image, dtype=np.float32)),
        "adv_image": save_image_png(str(case_dir / "adv.png"), np.asarray(attacked.image, dtype=np.float32)),
    }
    if defended is not None:
        refs["defended_image"] = save_image_png(
            str(case_dir / "defended.png"),
            np.asarray(defended.image, dtype=np.float32),
        )
    return refs


# 组装 `图文检索 案例 证据包 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _vlr_case_bundle_payload(
    *,
    row: dict[str, Any],
    cfg: AppConfig,
    model_tag: str,
    refs: dict[str, str],
    delta_metrics: tuple[int, float, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean = row["clean"]
    attacked = row["attacked"]
    defended = row.get("defended")
    l0, l2, linf = delta_metrics
    clean_score = row.get("clean_score")
    adv_score = row.get("adv_score")
    defended_score = row.get("defended_score")
    score_drop = float(row.get("score_drop") or 0.0)
    defense_gain = float((defended_score or adv_score or 0.0) - (adv_score or 0.0)) if defended_score is not None else 0.0
    text_diff = 0.0 if str(clean.text) == str(attacked.text) else 1.0
    clean_text = str(clean.text or "")
    attacked_text = str(attacked.text or "")
    defended_text = str(defended.text or "") if defended is not None else ""
    success = bool(score_drop > 0.0)
    bundle = {
        "task_kind": "vlr",
        "sample": {
            "sample_id": str(clean.sample_id),
            "text": clean_text,
            "target_text": str(clean.target_text or ""),
            "metadata": dict(clean.metadata),
        },
        "adversarial": {
            "sample_id": str(attacked.sample_id),
            "text": attacked_text,
            "perturbation_l0": l0,
            "perturbation_l2": l2,
            "perturbation_linf": linf,
            "metadata": dict(attacked.metadata),
        },
        "defended": {
            "sample_id": str(defended.sample_id) if defended is not None else "",
            "text": defended_text,
            "metadata": dict(defended.metadata) if defended is not None else {},
        },
        "inputs": {"clean": {"text": clean_text}, "adv": {"text": attacked_text}, "defended": {"text": defended_text}},
        "dataset_tag": str(cfg.dataset.benchmark_tag or cfg.dataset.kind),
        "model_tag": model_tag,
        "outputs": {
            "clean": {"text": _as_case_score_text(clean_score), "score": clean_score},
            "adv": {"text": _as_case_score_text(adv_score), "score": adv_score},
            "defended": {"text": _as_case_score_text(defended_score), "score": defended_score},
        },
        "metrics": {
            "score_drop": score_drop,
            "defense_gain": defense_gain,
            "text_diff_score": text_diff,
            "perturbation_l2": l2,
            "perturbation_linf": linf,
        },
        "judge": {
            "success": success,
            "reason": "paired_similarity_decreased" if success else "paired_similarity_not_decreased",
        },
        "diagnostics": {
            "clean_score": clean_score,
            "adv_score": adv_score,
            "defended_score": defended_score,
            "text_diff_score": text_diff,
            "embedding_shift": abs(score_drop),
        },
        "artifact_refs": refs,
        "visual_labels": {"clean": "原始图像", "adv": "对抗图像", "defended": "防御图像"},
    }
    summary = {"success": success, "perturbation_l2": l2, "perturbation_linf": linf, "defense_gain": defense_gain}
    return bundle, summary


# 写出 `图文检索 案例 证据包`，保证后续报告、页面或复现实验能读取。
def _write_vlr_case_bundle(
    *,
    case_dir: Path,
    row: dict[str, Any],
    cfg: AppConfig,
    model_tag: str,
) -> dict[str, Any]:
    clean = row["clean"]
    attacked = row["attacked"]
    defended = row.get("defended")
    refs = _save_vlr_case_images(case_dir, clean, attacked, defended)
    bundle, summary = _vlr_case_bundle_payload(
        row=row,
        cfg=cfg,
        model_tag=model_tag,
        refs=refs,
        delta_metrics=_sample_delta_metrics(clean, attacked),
    )
    refs["case_bundle"] = write_json(str(case_dir / "case_bundle.json"), bundle)
    return {
        "sample_id": str(clean.sample_id),
        "case_dir": str(case_dir),
        "judge_success": summary["success"],
        "perturbation_l2": summary["perturbation_l2"],
        "perturbation_linf": summary["perturbation_linf"],
        "defense_gain_sample": summary["defense_gain"],
    }


# 写出 `图文检索 案例 bundles`，保证后续报告、页面或复现实验能读取。
def _write_vlr_case_bundles(cfg: AppConfig, setup: dict[str, Any], state: dict[str, Any]) -> str:
    # Report pages need representative VLR case bundles even when the sample-store
    # feature is disabled; this keeps retrieval evidence aligned with VQA/Caption.
    if str(state.get("eval_scope")) == "clean":
        return ""
    clean_samples = list(state.get("clean_samples_subset") or [])
    attacked_samples = list(state.get("attacked_samples_subset") or [])
    if not clean_samples or not attacked_samples:
        return ""

    run_dir = Path(str(setup["run_dir"]))
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    max_cases = int(getattr(cfg.report, "top_k_cases", 0) or 0)
    rows = _case_bundle_rows(
        adapter=setup["surrogate_adapter"],
        clean_samples=clean_samples,
        attacked_samples=attacked_samples,
        defended_samples=list(state.get("defended_attack_samples") or []),
        max_cases=max_cases,
    )
    index_rows = [
        _write_vlr_case_bundle(
            case_dir=cases_dir / _sanitize_dir_name(str(row["sample_id"])),
            row=row,
            cfg=cfg,
            model_tag=str(setup["surrogate_name"]),
        )
        for row in rows
    ]
    return write_jsonl(str(run_dir / "cases_index.jsonl"), index_rows) if index_rows else ""


# 计算 `图文检索 stage 指标`，把原始模型输出汇总成页面和报告使用的指标字段。
def _vlr_stage_metrics(victim_metrics: dict[str, Any], victim_names: list[str]) -> dict[str, dict[str, Any]]:
    return {
        "clean": {name: victim_metrics.get(name, {}).get("clean", {}) for name in victim_names},
        "attacked": {name: victim_metrics.get(name, {}).get("attacked", {}) for name in victim_names},
        "conditional": {name: victim_metrics.get(name, {}).get("conditional", {}) for name in victim_names},
        "defended_attack": {name: victim_metrics.get(name, {}).get("defended_attack", {}) for name in victim_names},
        "defended_clean": {name: victim_metrics.get(name, {}).get("defended_clean", {}) for name in victim_names},
    }


# 写出 `图文检索 recall plots`，保证后续报告、页面或复现实验能读取。
def _write_vlr_recall_plots(*, run_dir: str, victim_metrics: dict[str, Any], victim_names: list[str], ks: list[int]) -> None:
    for victim_name in victim_names:
        clean_m = victim_metrics.get(victim_name, {}).get("clean", {}) or {}
        adv_m = victim_metrics.get(victim_name, {}).get("attacked", {}) or {}
        def_m = victim_metrics.get(victim_name, {}).get("defended_attack", {}) or {}
        labels: list[str] = []
        clean_vals: list[float] = []
        adv_vals: list[float] = []
        def_vals: list[float] = []
        for k in ks:
            labels.append(f"图检文前{k}")
            clean_vals.append(float(clean_m.get(f"ir_r@{k}", 0.0)))
            adv_vals.append(float(adv_m.get(f"ir_r@{k}", 0.0)))
            def_vals.append(float(def_m.get(f"ir_r@{k}", adv_m.get(f"ir_r@{k}", 0.0))))
        for k in ks:
            labels.append(f"文检图前{k}")
            clean_vals.append(float(clean_m.get(f"tr_r@{k}", 0.0)))
            adv_vals.append(float(adv_m.get(f"tr_r@{k}", 0.0)))
            def_vals.append(float(def_m.get(f"tr_r@{k}", adv_m.get(f"tr_r@{k}", 0.0))))
        plot_grouped_bar(
            labels=labels,
            series={"clean": clean_vals, "attacked": adv_vals, "defended": def_vals},
            out_path=f"{run_dir}/vlr_recall_{_sanitize_dir_name(victim_name)}.png",
            title=f"图文检索命中率（visual-language retrieval recall）({victim_name})",
            ylim=(0.0, 1.0),
        )


# 执行 `图文检索 transfer plot values` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _vlr_transfer_plot_values(victim_metrics: dict[str, Any], victim_names: list[str]) -> tuple[list[str], list[float], list[float]]:
    labels = [str(x) for x in victim_names]
    vals: list[float] = []
    vals_def: list[float] = []
    for victim_name in victim_names:
        adv_m = victim_metrics.get(victim_name, {}).get("attacked", {}) or {}
        def_m = victim_metrics.get(victim_name, {}).get("defended_attack", {}) or {}
        attacked = 0.5 * (float(adv_m.get("ir_asr@1", 0.0)) + float(adv_m.get("tr_asr@1", 0.0)))
        defended = 0.5 * (float(def_m.get("ir_asr@1", attacked)) + float(def_m.get("tr_asr@1", attacked)))
        vals.append(attacked)
        vals_def.append(defended)
    return labels, vals, vals_def


# 写出 `图文检索 transfer and 排名 plots`，保证后续报告、页面或复现实验能读取。
def _write_vlr_transfer_and_rank_plots(*, run_dir: str, victim_metrics: dict[str, Any], victim_names: list[str], defense: Any | None, asr_attack: float, asr_defended: float) -> None:
    labels, vals, vals_def = _vlr_transfer_plot_values(victim_metrics, victim_names)
    plot_grouped_bar(
        labels=labels,
        series={"攻击后首位攻击成功率": vals, "防御后首位攻击成功率": vals_def},
        out_path=f"{run_dir}/vlr_transfer_asr1.png",
        title="迁移攻击成功率（transfer attack success rate，首位平均）",
        ylim=(0.0, 1.0),
    )
    clean_rank, adv_rank = _vlr_rank_plot_values(victim_metrics, victim_names)
    plot_grouped_bar(
        labels=labels,
        series={"clean_mean_rank": clean_rank, "attacked_mean_rank": adv_rank},
        out_path=f"{run_dir}/vlr_mean_rank_compare.png",
        title="平均排名（mean rank，越低越好）",
        ylim=None,
    )
    if defense is not None:
        plot_stage_compare_bar(
            out_path=f"{run_dir}/vlr_stage_compare_asr.png",
            asr_attack=float(asr_attack),
            asr_defended=float(asr_defended),
            title="VLR Stage Compare",
        )
        plot_defense_recovery_curve(
            labels=labels,
            attacked_vals=vals,
            defended_vals=vals_def,
            out_path=f"{run_dir}/vlr_defense_recovery_curve.png",
            title="防御恢复效果（defense recovery，首位攻击成功率平均）",
        )


# 执行 `图文检索 排名 plot values` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _vlr_rank_plot_values(victim_metrics: dict[str, Any], victim_names: list[str]) -> tuple[list[float], list[float]]:
    clean_rank: list[float] = []
    adv_rank: list[float] = []
    for victim_name in victim_names:
        clean_m = victim_metrics.get(victim_name, {}).get("clean", {}) or {}
        adv_m = victim_metrics.get(victim_name, {}).get("attacked", {}) or {}
        clean_rank.append(0.5 * (float(clean_m.get("mean_rank_ir", 0.0)) + float(clean_m.get("mean_rank_tr", 0.0))))
        adv_rank.append(0.5 * (float(adv_m.get("mean_rank_ir", 0.0)) + float(adv_m.get("mean_rank_tr", 0.0))))
    return clean_rank, adv_rank


# 写出 `图文检索 plots`，保证后续报告、页面或复现实验能读取。
def _write_vlr_plots(*, cfg: AppConfig, run_dir: str, eval_scope: str, victim_metrics: dict[str, Any], victim_names: list[str], ks: list[int], outcome: dict[str, Any], defense: Any | None) -> None:
    if cfg.runner.save_plots and outcome["l2_values"]:
        plot_metric_curve(list(outcome["l2_values"]), "perturbation_l2", f"{run_dir}/metric_curve_l2.png")
    if cfg.runner.save_plots and outcome["linf_values"]:
        plot_metric_curve(list(outcome["linf_values"]), "perturbation_linf", f"{run_dir}/metric_curve_linf.png")
    if not (cfg.runner.save_plots and eval_scope != "clean"):
        return
    _write_vlr_recall_plots(run_dir=run_dir, victim_metrics=victim_metrics, victim_names=victim_names, ks=ks)
    _write_vlr_transfer_and_rank_plots(
        run_dir=run_dir,
        victim_metrics=victim_metrics,
        victim_names=victim_names,
        defense=defense,
        asr_attack=float(outcome["asr_attack"]),
        asr_defended=float(outcome["asr_defended"]),
    )


# 执行 `finish 图文检索 运行记录` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _finish_vlr_run(cfg: AppConfig, setup: dict[str, Any], state: dict[str, Any], progress: Callable[[str, str, float | None, str], None] | None) -> RunArtifacts:
    _emit_progress(progress, "result_aggregation", "running", 90, "正在汇总多模型指标、风险分数和样本级摘要。")
    run_id = str(setup["run_id"])
    run_dir = str(setup["run_dir"])
    victim_names = list(setup["victim_names"])
    victim_metrics = state["victim_metrics"]
    for victim_name in victim_names:
        victim_metrics.setdefault(victim_name, {})["status"] = state["victim_status"].get(victim_name, {})
    outcome = _vlr_outcome_context(
        cfg=cfg,
        surrogate_adapter=setup["surrogate_adapter"],
        clean_index=state["clean_index"],
        attacked_index=state["attacked_index"],
        clean_samples_subset=state["clean_samples_subset"],
        attacked_samples_subset=state["attacked_samples_subset"],
        victim_metrics=victim_metrics,
        victim_names=victim_names,
        eval_scope=state["eval_scope"],
        attack_debug=state["attack_debug"],
    )
    victim_compare = _victim_compare_rows(victim_metrics, victim_names, state["victim_status"])
    metric_quality = _metric_quality_report(victim_metrics, victim_names)
    defense_compare_rows = _defense_compare_rows(victim_metrics, victim_names, state["ks"])
    summary = _build_vlr_summary_payload(
        cfg=cfg,
        run_id=run_id,
        clean_index=state["clean_index"],
        surrogate_name=str(setup["surrogate_name"]),
        victim_names=victim_names,
        victim_metrics=victim_metrics,
        victim_status=state["victim_status"],
        victim_failures=state["victim_failures"],
        victim_compare=victim_compare,
        defense_compare_rows=defense_compare_rows,
        feature_projection=state["feature_projection"],
        attack_debug=state["attack_debug"],
        defended_attack_debug=state["defended_attack_debug"],
        defended_clean_debug=state["defended_clean_debug"],
        metric_quality=metric_quality,
        outcome=outcome,
        defense=setup["defense"],
        ks=state["ks"],
        eval_scope=state["eval_scope"],
    )
    results_path = write_results(run_dir, state["results_rows"])
    summary_path = write_summary(run_dir, summary)
    report_data = _build_vlr_report_payload(
        summary=summary,
        cfg=cfg,
        clean_index=state["clean_index"],
        victim_names=victim_names,
        victim_metrics=victim_metrics,
        victim_status=state["victim_status"],
        victim_failures=state["victim_failures"],
        victim_compare=victim_compare,
        defense_compare_rows=defense_compare_rows,
        failure_rows=state["failure_rows"],
        feature_projection=state["feature_projection"],
        metric_quality=metric_quality,
        outcome=outcome,
    )
    write_json_snapshot(run_dir, "report_data.json", report_data)
    run_index_path = _write_vlr_case_bundles(cfg, setup, state)
    _write_vlr_plots(cfg=cfg, run_dir=run_dir, eval_scope=state["eval_scope"], victim_metrics=victim_metrics, victim_names=victim_names, ks=state["ks"], outcome=outcome, defense=setup["defense"])
    _emit_progress(progress, "report_writing", "running", 97, "正在写入摘要、报告和图表文件。")
    report_path = write_report(run_dir, summary=summary, rows=state["failure_rows"])
    _emit_progress(progress, "report_writing", "success", 99, "运行报告写入完成。")
    return RunArtifacts(run_id=run_id, run_dir=run_dir, results_path=results_path, summary_path=summary_path, report_path=report_path, run_index_path=run_index_path, benchmark_summary_path="")


# 执行 `setup 图文检索 运行记录` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _setup_vlr_run(cfg: AppConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    run_id = new_run_id()
    run_dir = make_run_dir(cfg.artifacts_dir, run_id)
    write_json_snapshot(run_dir, "config_snapshot.json", asdict(cfg))
    write_env_snapshot(run_dir)
    surrogate_name, victim_names = _select_victim_names(cfg)
    LOG.info("VLR run: surrogate=%s victims=%s", surrogate_name, ",".join(victim_names))
    surrogate_adapter = create("model_adapter", surrogate_name)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "surrogate_name": surrogate_name,
        "victim_names": victim_names,
        "surrogate_adapter": surrogate_adapter,
        "victims": {name: create("model_adapter", name) for name in victim_names},
        "attack": create("attack", cfg.plugins.attack),
        "defense": create("defense", cfg.plugins.defense) if bool(cfg.defense.enabled) else None,
    }


# 执行 `execute 图文检索 stages` 辅助逻辑，保持评测运行器中的输入处理和结果输出一致。
def _execute_vlr_stages(cfg: AppConfig, setup: dict[str, Any], progress: Callable[[str, str, float | None, str], None] | None) -> dict[str, Any]:
    clean_samples_subset, clean_index = _load_vlr_samples_and_index(cfg, progress)
    ks = list(cfg.task.retrieval_k or [1, 5, 10])
    eval_scope = str(cfg.task.eval_scope or "clean")
    batch_size = int(cfg.attack.batch_size or 16)
    clean_sims, victim_metrics, results_rows, victim_status = _evaluate_clean_vlr_stage(
        victims=setup["victims"],
        victim_names=setup["victim_names"],
        clean_index=clean_index,
        ks=ks,
        batch_size=batch_size,
        progress=progress,
    )
    attack_state = _run_attacked_vlr_stage(
        cfg,
        {
            "run_dir": setup["run_dir"],
            "clean_samples_subset": clean_samples_subset,
            "clean_index": clean_index,
            "clean_sims": clean_sims,
            "victims": setup["victims"],
            "victim_names": setup["victim_names"],
            "surrogate_adapter": setup["surrogate_adapter"],
            "attack": setup["attack"],
            "defense": setup["defense"],
            "ks": ks,
            "eval_scope": eval_scope,
            "batch_size": batch_size,
            "victim_metrics": victim_metrics,
            "results_rows": results_rows,
            "victim_status": victim_status,
        },
        progress,
    )
    defended_clean_debug, defended_clean_index = _run_clean_defense_vlr_stage(
        cfg=cfg,
        run_dir=setup["run_dir"],
        clean_samples_subset=clean_samples_subset,
        victims=setup["victims"],
        victim_names=setup["victim_names"],
        surrogate_adapter=setup["surrogate_adapter"],
        defense=setup["defense"],
        ks=ks,
        batch_size=batch_size,
        progress=progress,
        victim_metrics=victim_metrics,
        results_rows=results_rows,
        victim_status=victim_status,
    )
    return _vlr_stage_state(
        clean_samples_subset=clean_samples_subset,
        clean_index=clean_index,
        ks=ks,
        eval_scope=eval_scope,
        victim_metrics=victim_metrics,
        results_rows=results_rows,
        victim_status=victim_status,
        attack_state=attack_state,
        defended_clean_debug=defended_clean_debug,
        defended_clean_index=defended_clean_index,
        surrogate_adapter=setup["surrogate_adapter"],
    )


# 判断或归一 `图文检索 stage state` 状态，让调用方可以稳定渲染能力和可用性。
def _vlr_stage_state(
    *,
    clean_samples_subset: list[Sample],
    clean_index: VLRIndex,
    ks: list[int],
    eval_scope: str,
    victim_metrics: dict[str, Any],
    results_rows: list[dict[str, Any]],
    victim_status: dict[str, dict[str, str]],
    attack_state: dict[str, Any],
    defended_clean_debug: dict[str, Any],
    defended_clean_index: VLRIndex | None,
    surrogate_adapter: Any,
) -> dict[str, Any]:
    feature_projection = _build_feature_projection(
        surrogate_adapter=surrogate_adapter,
        clean_index=clean_index,
        attacked_index=attack_state["attacked_index"],
        defended_attack_index=attack_state["defended_attack_index"],
        defended_clean_index=defended_clean_index,
        max_points_per_group=64,
    )
    failure_rows = _build_vlr_failure_rows(
        eval_scope=eval_scope,
        attacked_index=attack_state["attacked_index"],
        attacked_sims=attack_state["attacked_sims"],
        ks=ks,
    )
    return {
        "clean_samples_subset": clean_samples_subset,
        "clean_index": clean_index,
        "ks": ks,
        "eval_scope": eval_scope,
        "victim_metrics": victim_metrics,
        "results_rows": results_rows,
        "victim_status": victim_status,
        "victim_failures": [],
        "failure_rows": failure_rows,
        "feature_projection": feature_projection,
        "defended_clean_debug": defended_clean_debug,
        **attack_state,
    }


# 作为 `retrieval_runner.py` 的执行入口，串联参数读取、核心处理和退出状态。
def run(cfg: AppConfig, progress: Callable[[str, str, float | None, str], None] | None = None) -> RunArtifacts:
    """Run Vision-Language Retrieval (VLR) evaluation.

    Outputs:
      - results.jsonl: per-victim metrics rows (clean + attacked)
      - summary.json: aggregated UI-compatible fields + per-victim VLR metrics
      - report_data.json, report.html
    """
    setup = _setup_vlr_run(cfg)
    state = _execute_vlr_stages(cfg, setup, progress)
    artifacts = _finish_vlr_run(cfg, setup, state, progress)
    if bool(getattr(cfg.runner, "stop_local_vlm_after_run", False)):
        stop_local_vlm_servers(adapters=local_vlm_adapters(setup["victim_names"]))
        empty_cuda_cache()
    return artifacts
