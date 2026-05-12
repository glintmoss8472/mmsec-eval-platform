# 文件说明：该文件属于自动化测试，集中实现 test cli smoke 相关逻辑。
from pathlib import Path

from mmsec_eval.cli import main


# 中文注释：验证 test_cli_help_path 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_cli_help_path(tmp_path: Path):
    cfg = tmp_path / "cli_smoke.yaml"
    cfg.write_text(
        "\n".join(
            [
                "seed: 1",
                "artifacts_dir: '" + str(tmp_path / "artifacts").replace("\\", "/") + "'",
                "plugins:",
                "  model_adapter: clip_hf",
                "  attack: advedm",
                "  metric: basic",
                "  judge: rule",
                "dataset:",
                "  kind: toy_shapes",
                "  num_samples: 2",
                "  image_size: 64",
                "attack:",
                "  steps: 1",
                "  patch_size: 8",
                "runner:",
                "  max_samples: 2",
                "  continue_on_error: false",
            ]
        ),
        encoding="utf-8",
    )
    rc = main(["run-eval", "--config", str(cfg)])
    assert rc == 0
