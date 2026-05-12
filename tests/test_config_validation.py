# 文件说明：该文件属于自动化测试，集中实现 test config validation 相关逻辑。
from pathlib import Path

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.validate import validate_config
from mmsec_eval.plugins.builtin import register_builtin_plugins


# 中文注释：验证 test_validate_config_ok 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_validate_config_ok():
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    validate_config(cfg)


# 中文注释：验证 test_validate_config_allows_mini_flickr_default_root 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_validate_config_allows_mini_flickr_default_root():
    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")
    cfg.dataset.kind = "mini_flickr"
    cfg.dataset.root = ""
    cfg.dataset.image_dir = "images"
    cfg.dataset.captions_file = "captions_index.jsonl"
    cfg.dataset.split = "test"
    validate_config(cfg)
    assert Path(cfg.dataset.root).name == "mini_flickr"
    assert Path(cfg.dataset.root).parent.name == "data"

# 中文注释：验证 test_validate_config_rejects_invalid_attack_defense_and_runtime_values 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_validate_config_rejects_invalid_attack_defense_and_runtime_values():
    import pytest

    from mmsec_eval.exceptions import ConfigError

    register_builtin_plugins()
    cfg = load_config("configs/mvp.yaml")

    cfg.attack.steps = 0
    with pytest.raises(ConfigError, match="attack.steps must be > 0"):
        validate_config(cfg)

    cfg = load_config("configs/mvp.yaml")
    cfg.defense.median_kernel = 2
    with pytest.raises(ConfigError, match="defense.median_kernel must be odd"):
        validate_config(cfg)

    cfg = load_config("configs/mvp.yaml")
    cfg.runtime.device = "cpu"
    with pytest.raises(ConfigError, match="runtime.device='cpu' is not allowed"):
        validate_config(cfg)
