from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable


_ARTICLES = {"a", "an", "the"}
_NUM_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_answer(text: object) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"[\n\r\t]+", " ", value)
    value = re.sub(r"[^0-9a-z\u4e00-\u9fff\s]+", " ", value)
    tokens = []
    for token in value.split():
        mapped = _NUM_WORDS.get(token, token)
        if mapped in _ARTICLES:
            continue
        tokens.append(mapped)
    return " ".join(tokens).strip()


def answer_matches(candidate: object, answer: object, aliases: Iterable[object] = ()) -> bool:
    cand = normalize_answer(candidate)
    if not cand:
        return False
    accepted = [normalize_answer(answer), *[normalize_answer(item) for item in aliases]]
    accepted = [item for item in accepted if item]
    return any(cand == item or item in cand for item in accepted)


def yes_no_value(text: object) -> bool | None:
    norm = normalize_answer(text)
    if not norm:
        return None
    first = norm.split()[0]
    if first in {"yes", "true", "1", "是", "有", "存在"}:
        return True
    if first in {"no", "false", "0", "否", "没有", "不存在"}:
        return False
    return None


def token_set(text: object) -> set[str]:
    return {tok for tok in normalize_answer(text).split() if tok}


def text_similarity(a: object, b: object) -> float:
    sa = normalize_answer(a)
    sb = normalize_answer(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(SequenceMatcher(a=sa, b=sb).ratio())


def object_present(text: object, object_name: object, aliases: Iterable[object] = ()) -> bool:
    haystack = f" {normalize_answer(text)} "
    candidates = [normalize_answer(object_name), *[normalize_answer(item) for item in aliases]]
    for candidate in candidates:
        if not candidate:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", haystack):
            return True
    return False


def object_jaccard(clean_objects: Iterable[object], attacked_objects: Iterable[object]) -> float:
    clean = {normalize_answer(item) for item in clean_objects if normalize_answer(item)}
    attacked = {normalize_answer(item) for item in attacked_objects if normalize_answer(item)}
    if not clean and not attacked:
        return 1.0
    return float(len(clean & attacked) / max(1, len(clean | attacked)))
