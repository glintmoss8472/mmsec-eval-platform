from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from typing import Any

import numpy as np

from mmsec_eval.model_adapters.image_utils import image_to_pil_rgb


def encode_image_b64(image: np.ndarray, *, image_format: str = "PNG") -> str:
    pil = image_to_pil_rgb(image)
    buf = BytesIO()
    pil.save(buf, format=image_format)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_json_payload(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    parse_error = ""
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError as exc:
        parse_error = f"direct JSON parse failed: {type(exc).__name__}"

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError as exc:
            parse_error = f"embedded JSON parse failed: {type(exc).__name__}"
    payload = {"text": raw}
    if parse_error:
        payload["parse_error"] = parse_error
    return payload


def normalize_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(default)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        if score <= 10.0:
            return score / 10.0
        return 1.0
    return score
