from __future__ import annotations

from pathlib import Path

import pytest

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.attacks.advclip.train import train_advclip_patch
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import register
from mmsec_eval.runner.retrieval_runner import run as run_vlr
from mmsec_eval.types import ModelOutput, Sample


class _FailingAdapter:
    def predict(self, sample: Sample) -> ModelOutput:  # pragma: no cover - not used by retrieval runner
        raise RuntimeError("predict not supported")

    def score_pairs(self, pairs, batch_size: int = 8):
        raise RuntimeError("intentional scoring failure")


def test_vlr_victim_failure_is_strict(tmp_path: Path):
    register_builtin_plugins()
    register("model_adapter", "failing", lambda: _FailingAdapter())

    cfg = load_config("configs/mvp.yaml")
    cfg.artifacts_dir = str(tmp_path / "artifacts")
    cfg.task.kind = "vlr"
    cfg.task.eval_scope = "image"
    cfg.task.retrieval_k = [1, 5, 10]
    cfg.runner.surrogate_model_adapter = "clip_hf"
    cfg.runner.victim_model_adapters = ["clip_hf", "failing"]
    cfg.runner.max_pairs = 0
    cfg.plugins.attack = "advclip"
    cfg.dataset.kind = "toy_shapes"
    cfg.dataset.num_samples = 6
    cfg.runner.max_samples = 6
    cfg.runner.continue_on_error = False
    cfg.attack.patch_train_steps = 5

    validate_config(cfg)
    train_advclip_patch(cfg)
    with pytest.raises(RuntimeError):
        run_vlr(cfg)
