from __future__ import annotations

import re


def make_snippets(text: str, max_chars: int = 800) -> dict:
    clean = re.sub(r"\s+", " ", text or "").strip()
    head = clean[:max_chars]
    first_200 = clean[:200]
    return {
        "first_200": first_200,
        "snippet": head,
        "length": len(clean),
    }

