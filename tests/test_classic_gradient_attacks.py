from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.plugins.registry import create
from mmsec_eval.types import AttackContext, Sample


class DummyTorchAdapter:
    _device = "cpu"

    def score_pairs_torch(self, images_bchw, texts, output_attentions: bool = False):
        del output_attentions
        text_scale = torch.tensor([max(1.0, float(len(str(text).split()))) for text in texts], device=images_bchw.device)
        return images_bchw.mean(dim=(1, 2, 3)) * text_scale


def _cfg():
    attack = SimpleNamespace(
        mode="A",
        epsilon=0.08,
        step_size=0.02,
        steps=4,
        momentum_decay=1.0,
        diversity_prob=0.7,
        resize_rate=0.9,
        kernel_size=5,
        kernel_sigma=1.0,
        variance_samples=1,
        variance_radius=0.05,
        nesterov_scale=1.0,
        cw_const=0.1,
        cw_confidence=0.1,
    )
    task = SimpleNamespace(kind="vlr", eval_scope="image")
    return SimpleNamespace(attack=attack, task=task)


def _sample():
    image = np.full((32, 32, 3), 0.6, dtype=np.float32)
    return Sample(sample_id="classic-001", image=image, text="a bright object")


def test_classic_gradient_attacks_smoke(tmp_path):
    register_builtin_plugins()
    adapter = DummyTorchAdapter()
    ctx = AttackContext(
        config=_cfg(),
        model_adapter=adapter,
        surrogate_model_adapter=adapter,
        sample_debug_dir=str(tmp_path / "debug"),
    )

    for name in ["fgsm", "pgd", "mifgsm", "dtmifgsm", "cw"]:
        plugin = create("attack", name)
        attacked = plugin.attack(_sample(), ctx)
        assert attacked.sample.metadata["attack_name"] == name
        assert attacked.sample.metadata["attack_scope"] == "image"
        assert attacked.sample.metadata["attack_implementation"] in {"torchattacks", "builtin"}
        assert len(attacked.attack_trace) > 0
        assert attacked.perturbation_linf >= 0.0
        debug_path = attacked.metadata.get("debug_path", "")
        assert debug_path
        payload = json.loads(Path(debug_path).read_text(encoding="utf-8"))
        assert payload["implementation"] in {"torchattacks", "builtin"}


def test_classic_gradient_batch_splits_mixed_image_shapes(tmp_path):
    register_builtin_plugins()
    adapter = DummyTorchAdapter()
    ctx = AttackContext(
        config=_cfg(),
        model_adapter=adapter,
        surrogate_model_adapter=adapter,
        sample_debug_dir=str(tmp_path / "debug"),
    )
    sample_a = _sample()
    sample_b = Sample(
        sample_id="classic-002",
        image=np.full((24, 40, 3), 0.5, dtype=np.float32),
        text="a compact object",
    )

    plugin = create("attack", "fgsm")
    attacked = plugin.attack_batch([sample_a, sample_b], ctx)

    assert len(attacked) == 2
    assert attacked[0].sample.image.shape == sample_a.image.shape
    assert attacked[1].sample.image.shape == sample_b.image.shape
    assert attacked[0].metadata.get("shape_grouped_batch") is True
    assert attacked[1].metadata.get("shape_grouped_batch") is True
    assert attacked[0].sample.metadata["attack_implementation"] == "builtin_batch"
