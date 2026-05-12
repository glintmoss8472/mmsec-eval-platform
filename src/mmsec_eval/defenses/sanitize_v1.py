from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from mmsec_eval.attacks.text_utils import run_text_repair
from mmsec_eval.plugins.base import DefensePlugin
from mmsec_eval.types import DefenseContext, DefendedSample, Sample


def _to_u8_hwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("sanitize_v1 expects HWC image with 3 channels")
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_f32_hwc(image_u8: np.ndarray) -> np.ndarray:
    return np.asarray(image_u8, dtype=np.float32) / 255.0


def _normalize_text(text: str) -> str:
    x = str(text or "").lower()
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _dedupe_tokens(text: str) -> str:
    tokens = [tok for tok in str(text or "").split() if tok]
    if not tokens:
        return ""
    out: list[str] = []
    for tok in tokens:
        if not out or out[-1] != tok:
            out.append(tok)
    return " ".join(out)


def _token_delta_ratio(src: str, dst: str) -> float:
    src_tokens = [tok for tok in str(src or "").split() if tok]
    dst_tokens = [tok for tok in str(dst or "").split() if tok]
    denom = max(1, len(src_tokens), len(dst_tokens))
    changed = abs(len(src_tokens) - len(dst_tokens))
    for a, b in zip(src_tokens, dst_tokens):
        if a != b:
            changed += 1
    return float(changed) / float(denom)


def _mean_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        return 0.0
    return float(np.mean(np.abs(arr_a - arr_b)))


def _apply_recipe(
    pil: Image.Image,
    *,
    resize_ratio: float,
    bit_depth: int,
    jpeg_quality: int,
    blur_sigma: float,
    median_kernel: int,
) -> Image.Image:
    out = pil.copy()
    orig_w, orig_h = out.size

    if resize_ratio < 1.0:
        down_w = max(16, int(round(orig_w * resize_ratio)))
        down_h = max(16, int(round(orig_h * resize_ratio)))
        out = out.resize((down_w, down_h), resample=Image.BILINEAR).resize((orig_w, orig_h), resample=Image.BILINEAR)

    if median_kernel >= 3:
        out = out.filter(ImageFilter.MedianFilter(size=int(median_kernel)))

    levels = max(2, 2**int(bit_depth))
    arr = np.asarray(out, dtype=np.float32)
    arr = np.round(arr / 255.0 * (levels - 1.0)) / (levels - 1.0)
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    out = Image.fromarray(arr)

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=int(jpeg_quality))
    buf.seek(0)
    out = Image.open(buf).convert("RGB")

    if blur_sigma > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
    return out


def _image_recipes(cfg) -> list[dict[str, float | int | str]]:
    strong_ratio = min(float(cfg.resize_ratio), float(cfg.strong_resize_ratio))
    strong_quality = min(int(cfg.jpeg_quality), int(cfg.strong_jpeg_quality))
    strong_bits = min(int(cfg.bit_depth), int(cfg.strong_bit_depth))
    strong_blur = max(float(cfg.blur_sigma), float(cfg.strong_blur_sigma))
    median_kernel = int(cfg.median_kernel)
    return [
        {
            "name": "identity",
            "resize_ratio": 1.0,
            "bit_depth": 8,
            "jpeg_quality": 100,
            "blur_sigma": 0.0,
            "median_kernel": 0,
        },
        {
            "name": "mild",
            "resize_ratio": float(cfg.resize_ratio),
            "bit_depth": int(cfg.bit_depth),
            "jpeg_quality": int(cfg.jpeg_quality),
            "blur_sigma": float(cfg.blur_sigma),
            "median_kernel": median_kernel,
        },
        {
            "name": "denoise",
            "resize_ratio": min(0.84, (float(cfg.resize_ratio) + strong_ratio) / 2.0),
            "bit_depth": max(2, min(int(cfg.bit_depth), strong_bits + 1)),
            "jpeg_quality": min(72, max(strong_quality, int(cfg.strong_jpeg_quality))),
            "blur_sigma": max(float(cfg.blur_sigma), 1.1),
            "median_kernel": max(3, median_kernel),
        },
        {
            "name": "robust",
            "resize_ratio": strong_ratio,
            "bit_depth": strong_bits,
            "jpeg_quality": strong_quality,
            "blur_sigma": strong_blur,
            "median_kernel": max(3, median_kernel if median_kernel % 2 == 1 else median_kernel + 1),
        },
    ]


def _text_candidates(text: str) -> list[str]:
    normalized = _normalize_text(text)
    punctuation_clean = re.sub(r"[^0-9a-z\s]+", " ", normalized)
    variants = [
        normalized,
        _normalize_text(punctuation_clean),
        _normalize_text(_dedupe_tokens(normalized)),
        str(text or ""),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _score_pair(adapter, image: np.ndarray, text: str) -> float:
    if adapter is None or not hasattr(adapter, "score_pairs"):
        return 0.0
    try:
        return float(adapter.score_pairs([(np.asarray(image, dtype=np.float32), str(text))], batch_size=1)[0])
    except (AttributeError, TypeError, ValueError, IndexError, RuntimeError):
        return 0.0


def _defense_config_snapshot(cfg) -> dict[str, object]:
    return {
        "resize_ratio": float(cfg.resize_ratio),
        "bit_depth": int(cfg.bit_depth),
        "jpeg_quality": int(cfg.jpeg_quality),
        "blur_sigma": float(cfg.blur_sigma),
        "median_kernel": int(cfg.median_kernel),
        "strong_resize_ratio": float(cfg.strong_resize_ratio),
        "strong_bit_depth": int(cfg.strong_bit_depth),
        "strong_jpeg_quality": int(cfg.strong_jpeg_quality),
        "strong_blur_sigma": float(cfg.strong_blur_sigma),
        "selection_penalty": float(cfg.selection_penalty),
        "text_normalize": bool(cfg.text_normalize),
        "text_repair": bool(cfg.text_repair),
        "text_repair_max_edits": int(cfg.text_repair_max_edits),
        "text_candidates_k": int(cfg.text_candidates_k),
    }


def _maybe_repair_text(*, image: np.ndarray, text: str, adapter, cfg) -> tuple[str, dict[str, object]]:
    if not (bool(cfg.text_repair) and hasattr(adapter, "score_pairs") and text):
        return str(text), {"method": "noop", "reason": "disabled"}
    repaired_text, debug = run_text_repair(
        image=np.asarray(image, dtype=np.float32),
        text=text,
        adapter=adapter,
        max_edits=int(cfg.text_repair_max_edits),
        candidates_k=int(cfg.text_candidates_k),
        prefer_mlm=True,
    )
    return str(repaired_text), dict(debug)


def _select_text_variant(
    *,
    image: np.ndarray,
    base_text: str,
    variants: list[str],
    adapter,
    selection_penalty: float,
) -> tuple[str, float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    best_text = base_text
    best_score = -1e18
    for candidate in variants:
        raw_score = _score_pair(adapter, np.asarray(image, dtype=np.float32), candidate)
        adjusted = float(raw_score) - selection_penalty * _token_delta_ratio(base_text, candidate)
        row = {
            "text": candidate,
            "raw_score": float(raw_score),
            "adjusted_score": float(adjusted),
            "delta_ratio": float(_token_delta_ratio(base_text, candidate)),
        }
        rows.append(row)
        if adjusted > best_score:
            best_score = adjusted
            best_text = candidate
    return best_text, float(best_score), rows


def _select_image_variant(
    *,
    pil: Image.Image,
    original_image: np.ndarray,
    text: str,
    adapter,
    cfg,
    selection_penalty: float,
) -> tuple[np.ndarray, str, float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    best_image = np.asarray(original_image, dtype=np.float32)
    best_score = -1e18
    best_recipe = "identity"
    for recipe in _image_recipes(cfg):
        candidate_pil = _apply_recipe(
            pil,
            resize_ratio=float(recipe["resize_ratio"]),
            bit_depth=int(recipe["bit_depth"]),
            jpeg_quality=int(recipe["jpeg_quality"]),
            blur_sigma=float(recipe["blur_sigma"]),
            median_kernel=int(recipe["median_kernel"]),
        )
        candidate = _to_f32_hwc(np.asarray(candidate_pil, dtype=np.uint8))
        raw_score = _score_pair(adapter, candidate, text)
        delta = _mean_abs_delta(original_image, candidate)
        adjusted = float(raw_score) - selection_penalty * float(delta)
        row = {
            "name": str(recipe["name"]),
            "raw_score": float(raw_score),
            "adjusted_score": float(adjusted),
            "image_delta": float(delta),
            "resize_ratio": float(recipe["resize_ratio"]),
            "bit_depth": int(recipe["bit_depth"]),
            "jpeg_quality": int(recipe["jpeg_quality"]),
            "blur_sigma": float(recipe["blur_sigma"]),
            "median_kernel": int(recipe["median_kernel"]),
        }
        rows.append(row)
        if adjusted > best_score:
            best_score = adjusted
            best_image = candidate
            best_recipe = str(recipe["name"])
    return best_image, best_recipe, float(best_score), rows


def _write_defense_trace(sample_debug_dir: str, trace: dict[str, object]) -> dict[str, str]:
    if not sample_debug_dir:
        return {}
    p = Path(sample_debug_dir)
    p.mkdir(parents=True, exist_ok=True)
    trace_path = p / "defense_trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"defense_trace": str(trace_path)}


def _make_defended_sample(*, sample: Sample, image: np.ndarray, text: str, stage: str) -> Sample:
    defended = Sample(
        sample_id=str(sample.sample_id),
        image=np.asarray(image, dtype=np.float32),
        text=str(text),
        target_text=str(sample.target_text or ""),
        metadata=dict(sample.metadata),
    )
    defended.metadata["defense_name"] = "sanitize_v1"
    defended.metadata["defense_stage"] = str(stage)
    return defended


class SanitizeDefenseV1(DefensePlugin):
    """Deterministic input sanitization defense.

    Pipeline: multi-strength image sanitization -> text normalization/repair -> score-guided selection.
    """

    def defend(self, sample: Sample, ctx: DefenseContext) -> DefendedSample:
        cfg = ctx.config.defense

        image_u8 = _to_u8_hwc(sample.image)
        pil = Image.fromarray(image_u8)

        trace: dict[str, object] = {
            "defense": "sanitize_v1",
            "stage": str(ctx.stage),
            "config": _defense_config_snapshot(cfg),
        }

        text_variants = _text_candidates(sample.text if bool(cfg.text_normalize) else str(sample.text))
        base_text = text_variants[0] if text_variants else str(sample.text or "")
        repaired_text, text_repair_debug = _maybe_repair_text(
            image=np.asarray(sample.image, dtype=np.float32),
            text=base_text,
            adapter=ctx.model_adapter,
            cfg=cfg,
        )
        if repaired_text and repaired_text not in text_variants:
            text_variants.append(str(repaired_text))

        selection_penalty = float(cfg.selection_penalty)
        best_text, best_text_score, text_candidates = _select_text_variant(
            image=np.asarray(sample.image, dtype=np.float32),
            base_text=base_text,
            variants=text_variants,
            adapter=ctx.model_adapter,
            selection_penalty=selection_penalty,
        )
        best_image, best_recipe, best_image_score, image_candidates = _select_image_variant(
            pil=pil,
            original_image=np.asarray(sample.image, dtype=np.float32),
            text=best_text,
            adapter=ctx.model_adapter,
            cfg=cfg,
            selection_penalty=selection_penalty,
        )

        repaired_best, repaired_best_debug = _maybe_repair_text(
            image=np.asarray(best_image, dtype=np.float32),
            text=best_text,
            adapter=ctx.model_adapter,
            cfg=cfg,
        )
        if repaired_best and repaired_best != best_text:
            repaired_score = _score_pair(ctx.model_adapter, np.asarray(best_image, dtype=np.float32), repaired_best)
            current_score = _score_pair(ctx.model_adapter, np.asarray(best_image, dtype=np.float32), best_text)
            if repaired_score > current_score + 1e-3:
                best_text = str(repaired_best)
                text_repair_debug = {
                    "first_pass": text_repair_debug,
                    "second_pass": repaired_best_debug,
                }

        adv = np.asarray(best_image, dtype=np.float32)
        text_out = str(best_text or base_text)
        trace["text_candidates"] = text_candidates
        trace["text_repair"] = text_repair_debug
        trace["image_candidates"] = image_candidates
        trace["selection"] = {
            "selected_recipe": best_recipe,
            "selected_text": text_out,
            "selected_image_score": float(best_image_score),
            "selected_text_score": float(best_text_score),
        }
        return DefendedSample(
            sample=_make_defended_sample(sample=sample, image=adv, text=text_out, stage=str(ctx.stage)),
            metadata={"trace": trace},
            artifact_refs=_write_defense_trace(str(ctx.sample_debug_dir or ""), trace),
        )
