from __future__ import annotations

import math

import numpy as np

from mmsec_eval.metrics.ssim_metric import ssim_simple
from mmsec_eval.plugins.base import MetricPlugin
from mmsec_eval.types import EvalRecord


def _text_similarity(a: str, b: str) -> float:
    sa = set((a or "").lower().split())
    sb = set((b or "").lower().split())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb) + 1e-8
    return float(inter / union)


def _psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(((x - y) ** 2).mean())
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def _safe_image_pair(record: EvalRecord) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(record.sample.image, dtype=np.float32)
    adv = np.asarray(record.attacked.sample.image, dtype=np.float32)
    if clean.shape != adv.shape:
        h = min(clean.shape[0], adv.shape[0])
        w = min(clean.shape[1], adv.shape[1])
        c = min(clean.shape[2], adv.shape[2])
        clean = clean[:h, :w, :c]
        adv = adv[:h, :w, :c]
    return clean, adv


class BasicMetrics(MetricPlugin):
    def compute(self, record: EvalRecord) -> dict[str, float]:
        try:
            clean, adv = _safe_image_pair(record)
            delta = adv - clean
            l0 = float(np.count_nonzero(np.abs(delta) > 1e-8))
            l2 = float(np.sqrt((delta * delta).sum()))
            linf = float(np.abs(delta).max()) if delta.size else 0.0
            sim = _text_similarity(record.pred_clean.text, record.pred_adv.text)
            ssim = float(ssim_simple(clean, adv))
            psnr = float(_psnr(clean, adv))
        except (AttributeError, TypeError, ValueError, IndexError):
            l0 = float(record.attacked.perturbation_l0)
            l2 = float(record.attacked.perturbation_l2)
            linf = float(record.attacked.perturbation_linf)
            sim = 0.0
            ssim = 0.0
            psnr = 0.0

        judge_success = float(record.judge.success) if record.judge else 0.0
        transfer = float(record.attacked.metadata.get("transfer_success", judge_success))

        return {
            "perturbation_l0": l0,
            "perturbation_l2": l2,
            "perturbation_linf": linf,
            "semantic_similarity": sim,
            "ssim": ssim,
            "psnr": psnr,
            "transfer_success": transfer,
        }
