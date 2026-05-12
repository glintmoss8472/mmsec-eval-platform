from mmsec_eval.config.loader import load_config


def test_load_config_returns_object():
    cfg = load_config("configs/mvp.yaml")
    assert cfg.seed >= 0
    assert cfg.plugins.attack


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
