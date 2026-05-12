from __future__ import annotations

import numpy as np

from mmsec_eval.attacks.text_utils import run_text_repair, run_text_replacement_attack


class _ScoredTextAdapter:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score_pairs(self, pairs, batch_size: int = 1):
        del batch_size
        return np.asarray([self._scores[str(text)] for _, text in pairs], dtype=np.float32)


def test_text_attack_fallback_drops_token_that_most_reduces_score():
    adapter = _ScoredTextAdapter(
        {
            "red square noisy": 0.9,
            "square noisy": 0.8,
            "red noisy": 0.6,
            "red square": 0.1,
        }
    )

    text, debug = run_text_replacement_attack(
        image=np.zeros((4, 4, 3), dtype=np.float32),
        text="red square noisy",
        adapter=adapter,
        eps_t=1,
        candidates_k=4,
        prefer_mlm=False,
    )

    assert text == "red square"
    assert debug["method"] == "token_drop_fallback"
    assert debug["eps_t"] == 1
    assert debug["num_edits"] == 1
    assert debug["edits"][0]["old_token"] == "noisy"
    assert debug["score_new"] < debug["score_orig"]


def test_text_repair_fallback_drops_token_that_most_increases_score():
    adapter = _ScoredTextAdapter(
        {
            "red square wrong": 0.2,
            "square wrong": 0.1,
            "red wrong": 0.3,
            "red square": 0.8,
        }
    )

    text, debug = run_text_repair(
        image=np.zeros((4, 4, 3), dtype=np.float32),
        text="red square wrong",
        adapter=adapter,
        max_edits=1,
        candidates_k=4,
        prefer_mlm=False,
    )

    assert text == "red square"
    assert debug["method"] == "token_drop_repair"
    assert debug["max_edits"] == 1
    assert debug["num_edits"] == 1
    assert debug["edits"][0]["old_token"] == "wrong"
    assert debug["score_new"] > debug["score_orig"]
