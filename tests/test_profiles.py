# 文件说明：该文件属于自动化测试，集中实现 test profiles 相关逻辑。
from mmsec_eval.config.loader import load_config


# 中文注释：验证 test_profile_gpu_full_loads 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_profile_gpu_full_loads():
    cfg = load_config("configs/profiles/gpu_full.yaml")
    assert cfg.runtime.device == "cuda"
    assert cfg.attack.mode == "B"


# 中文注释：验证 test_profile_cpu_smoke_loads 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_profile_cpu_smoke_loads():
    cfg = load_config("configs/profiles/cpu_smoke.yaml")
    assert cfg.runtime.device == "cuda"
    assert cfg.dataset.num_samples == 8
