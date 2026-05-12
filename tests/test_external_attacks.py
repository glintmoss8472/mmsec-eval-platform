# 文件说明：该文件属于自动化测试，集中实现 test external attacks 相关逻辑。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from mmsec_eval.attacks.external import ExternalAttackPrerequisiteError, FOAAttack, MPCAttackPlugin, VQAVisualCorruptionAttack, XTransferUAPAttack
from mmsec_eval.config.schema import AppConfig
from mmsec_eval.types import AttackContext, Sample


# 执行 `样本` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _sample() -> Sample:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[:, :16, 0] = 0.8
    image[:, 16:, 1] = 0.6
    return Sample(sample_id="s1", image=image, text="red and green blocks", target_text="target caption")


# 执行 `ctx` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _ctx(tmp_path: Path, cfg: AppConfig | None = None) -> AttackContext:
    cfg = cfg or AppConfig()
    return AttackContext(config=cfg, model_adapter=None, run_dir=str(tmp_path), sample_debug_dir=str(tmp_path / "debug"))


# 验证 `视觉问答 visual corruption outputs required metadata` 场景，防止相关行为在后续修改中退化。
def test_vqa_visual_corruption_outputs_required_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "VQA-Visual-Robustness-Benchmark"
    repo.mkdir()
    repo.joinpath("generator.py").write_text(
        "import numpy as np\n"
        "class Generator:\n"
        "    def __init__(self, dataset, logger):\n"
        "        self.dataset = dataset\n"
        "        self.logger = logger\n"
        "        self.validTransformations = {'Gaussian-noise_L3': (self.transformToGaussianNoise, 3)}\n"
        "    def transformToGaussianNoise(self, idx, severity=1):\n"
        "        image, *_ = self.dataset[idx]\n"
        "        return np.clip(image.astype(np.float32) + severity * 12.0, 0, 255)\n",
        encoding="utf-8",
    )
    cfg = AppConfig()
    cfg.attack.repo_dir = str(repo)
    cfg.attack.python_bin = sys.executable
    cfg.attack.command_template = (
        "{python} {project_root}/scripts/external_attacks/vqa_visual_corruption_one.py "
        "--repo_dir {repo_dir} --input_image {input_image} --output_image {output_image} "
        "--corruption_type {corruption_type} --severity {severity} --seed {seed} --trace_json {external_trace}"
    )
    cfg.attack.corruption_type = "gaussian_noise"
    cfg.attack.severity = 3
    cfg.attack.corruption_seed = 123

    attacked = VQAVisualCorruptionAttack().attack(_sample(), _ctx(tmp_path, cfg))

    assert attacked.metadata["attack_name"] == "vqa_visual_corruption"
    assert attacked.metadata["attack_mode"] == "official_visual_corruption"
    assert attacked.metadata["attack_scope"] == "image"
    assert attacked.metadata["perturbation_l0"] > 0
    assert attacked.metadata["perturbation_l2"] == pytest.approx(attacked.perturbation_l2)
    trace = attacked.metadata["attack_trace"][0]
    assert trace["repo_dir"] == str(repo.resolve())
    assert trace["returncode"] == 0
    assert trace["corruption_type"] == "gaussian_noise"
    assert trace["official_transformation"] == "Gaussian-noise_L3"
    assert trace["severity"] == 3
    assert trace["seed"] == 123
    assert trace["actual_params"]["official_corruption"] == "Gaussian-noise"
    assert Path(trace["output_image"]).exists()


# 验证 `视觉问答 visual corruption missing official repo raises` 场景，防止相关行为在后续修改中退化。
def test_vqa_visual_corruption_missing_official_repo_raises(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.attack.repo_dir = str(tmp_path / "missing_vqa_repo")
    cfg.attack.command_template = "{python} official.py {input_image} {output_image}"

    with pytest.raises(ExternalAttackPrerequisiteError, match="repo_dir not found"):
        VQAVisualCorruptionAttack().attack(_sample(), _ctx(tmp_path, cfg))


# 验证 `external 攻击 missing repo raises instead of returning unavailable` 场景，防止相关行为在后续修改中退化。
def test_external_attack_missing_repo_raises_instead_of_returning_unavailable(tmp_path: Path) -> None:
    cfg = AppConfig()
    cfg.attack.repo_dir = str(tmp_path / "missing_repo")
    cfg.attack.target_text = "make the image look like a target"
    cfg.attack.command_template = "{python} missing.py {input_image} {output_image}"

    with pytest.raises(ExternalAttackPrerequisiteError, match="repo_dir not found"):
        FOAAttack().attack(_sample(), _ctx(tmp_path, cfg))


# 验证 `external 攻击 mock success output 图像` 场景，防止相关行为在后续修改中退化。
def test_external_attack_mock_success_output_image(tmp_path: Path) -> None:
    repo = tmp_path / "mock_foa"
    repo.mkdir()
    script = repo / "mock_attack.py"
    script.write_text(
        "from PIL import Image, ImageEnhance\n"
        "import sys\n"
        "im = Image.open(sys.argv[1]).convert('RGB')\n"
        "ImageEnhance.Brightness(im).enhance(0.5).save(sys.argv[2])\n",
        encoding="utf-8",
    )
    cfg = AppConfig()
    cfg.attack.repo_dir = str(repo)
    cfg.attack.python_bin = sys.executable
    cfg.attack.target_text = "target caption"
    cfg.attack.command_template = "{python} mock_attack.py {input_image} {output_image}"
    cfg.attack.timeout_sec = 30

    attacked = FOAAttack().attack(_sample(), _ctx(tmp_path, cfg))

    assert attacked.metadata["attack_name"] == "foa_attack"
    assert attacked.metadata["attack_mode"] == "targeted_transfer"
    assert attacked.perturbation_l2 > 0
    trace = attacked.metadata["attack_trace"][0]
    assert trace["status"] == "success"
    assert trace["returncode"] == 0
    assert Path(trace["output_image"]).exists()


# 验证 `xtransfer official zoo repo adapter mock success` 场景，防止相关行为在后续修改中退化。
def test_xtransfer_official_zoo_repo_adapter_mock_success(tmp_path: Path) -> None:
    repo = tmp_path / "XTransferBench"
    zoo_dir = repo / "src" / "XTransferBench" / "zoo"
    zoo_dir.mkdir(parents=True)
    (repo / "src" / "XTransferBench" / "__init__.py").write_text("", encoding="utf-8")
    zoo_dir.joinpath("__init__.py").write_text(
        "import torch\n"
        "class FakeAttacker(torch.nn.Module):\n"
        "    def interpolate_epsilon(self, epsilon): self.epsilon = float(epsilon)\n"
        "    def forward(self, images): return torch.clamp(images + 0.01, 0, 1)\n"
        "def load_attacker(threat_model, method_name): return FakeAttacker()\n"
        "def list_attacker(threat_model): return ['fake_uap']\n",
        encoding="utf-8",
    )
    cfg = AppConfig()
    cfg.attack.repo_dir = str(repo)
    cfg.attack.uap_path = ""
    cfg.attack.uap_name = "fake_uap"
    cfg.attack.threat_model = "linf_non_targeted"
    cfg.attack.epsilon = 0.02
    cfg.attack.device = "cpu"

    attacked = XTransferUAPAttack().attack(_sample(), _ctx(tmp_path, cfg))

    trace = attacked.metadata["attack_trace"][0]
    assert trace["used_official_zoo"] is True
    assert trace["used_package_zoo"] is True
    assert trace["official_repo_dir"].endswith("XTransferBench")
    assert attacked.perturbation_linf > 0


# 验证 `xtransfer 本地 uap 路径 applies perturbation` 场景，防止相关行为在后续修改中退化。
def test_xtransfer_local_uap_path_applies_perturbation(tmp_path: Path) -> None:
    uap_path = tmp_path / "uap.npy"
    np.save(uap_path, np.full((32, 32, 3), 0.01, dtype=np.float32))
    cfg = AppConfig()
    cfg.attack.uap_path = str(uap_path)
    cfg.attack.epsilon = 0.02

    attacked = XTransferUAPAttack().attack(_sample(), _ctx(tmp_path, cfg))

    assert attacked.metadata["attack_name"] == "xtransfer_uap"
    trace = attacked.metadata["attack_trace"][0]
    assert trace["used_local_uap_path"] is True
    assert trace["used_package_zoo"] is False
    assert attacked.perturbation_linf <= 0.0201


# 验证 `mpc 攻击 requires target 图像 even with target 文本` 场景，防止相关行为在后续修改中退化。
def test_mpc_attack_requires_target_image_even_with_target_text(tmp_path: Path) -> None:
    repo = tmp_path / "MPCAttack"
    repo.mkdir()
    cfg = AppConfig()
    cfg.attack.repo_dir = str(repo)
    cfg.attack.target_text = "target caption"
    cfg.attack.command_template = "{python} mock.py --input {input_image} --output {output_image}"

    with pytest.raises(ExternalAttackPrerequisiteError, match="requires target_image"):
        MPCAttackPlugin().attack(_sample(), _ctx(tmp_path, cfg))
