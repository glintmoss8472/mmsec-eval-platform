from mmsec_eval.config.loader import load_config


def test_profile_gpu_full_loads():
    cfg = load_config("configs/profiles/gpu_full.yaml")
    assert cfg.runtime.device == "cuda"
    assert cfg.attack.mode == "B"


def test_profile_cpu_smoke_loads():
    cfg = load_config("configs/profiles/cpu_smoke.yaml")
    assert cfg.runtime.device == "cuda"
    assert cfg.dataset.num_samples == 8
