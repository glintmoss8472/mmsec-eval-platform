# 文件说明：该文件属于自动化测试，集中实现 test runner vlr victim tolerance 相关逻辑。
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


# 实现 `_FailingAdapter.predict` 的对象行为，维护该类在自动化测试中的调用契约。
class _FailingAdapter:
    # 实现 _FailingAdapter.predict 的核心行为，维护自动化测试在该对象上的调用契约。
    def predict(self, sample: Sample) -> ModelOutput:  # pragma: no cover - not used by retrieval runner
        raise RuntimeError("predict not supported")

    # 计算 `pairs`，为指标、风险或调度决策提供数值依据。
    def score_pairs(self, pairs, batch_size: int = 8):
        raise RuntimeError("intentional scoring failure")


# 验证 `图文检索 victim failure 是否 strict` 场景，防止相关行为在后续修改中退化。
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
