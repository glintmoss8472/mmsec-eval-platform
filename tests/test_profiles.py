# 文件说明：该文件属于自动化测试，集中实现 test profiles 相关逻辑。
from mmsec_eval.config.loader import load_config


# 验证 `profile gpu full loads` 场景，防止相关行为在后续修改中退化。
def test_profile_gpu_full_loads():
    cfg = load_config("configs/profiles/gpu_full.yaml")
    assert cfg.runtime.device == "cuda"
    assert cfg.attack.mode == "B"


# 验证 `profile cpu smoke loads` 场景，防止相关行为在后续修改中退化。
def test_profile_cpu_smoke_loads():
    cfg = load_config("configs/profiles/cpu_smoke.yaml")
    assert cfg.runtime.device == "cuda"
    assert cfg.dataset.num_samples == 8
