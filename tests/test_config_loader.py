# 文件说明：该文件属于自动化测试，集中实现 test config loader 相关逻辑。
from mmsec_eval.config.loader import load_config


# 中文注释：验证 test_load_config_returns_object 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_load_config_returns_object():
    cfg = load_config("configs/mvp.yaml")
    assert cfg.seed >= 0
    assert cfg.plugins.attack


# 中文注释：验证 test_load_config_migrates_legacy_task_limits 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_load_config_migrates_legacy_task_limits(tmp_path):
    cfg_path = tmp_path / "legacy_task_limits.yaml"
    cfg_path.write_text(
        """
task:
  kind: vlr
  eval_scope: image
  max_pairs: 128
  max_samples: 16
runner:
  continue_on_error: false
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.task.kind == "vlr"
    assert cfg.task.eval_scope == "image"
    assert cfg.runner.max_pairs == 128
    assert cfg.runner.max_samples == 16


# 中文注释：验证 test_load_config_migrates_legacy_report_metadata 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_load_config_migrates_legacy_report_metadata(tmp_path):
    cfg_path = tmp_path / "legacy_report_metadata.yaml"
    cfg_path.write_text(
        """
report:
  task_name: 多模态模型综合安全测评
  note: 浏览器提交入口
  top_k_cases: 3
""",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_path))

    assert cfg.report.top_k_cases == 3
    assert cfg.extra["report_task_name"] == "多模态模型综合安全测评"
    assert cfg.extra["report_note"] == "浏览器提交入口"
