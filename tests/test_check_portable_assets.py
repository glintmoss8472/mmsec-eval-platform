from __future__ import annotations

import importlib.util
from pathlib import Path

from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_LOCAL_DIRS


def _load_check_portable_assets():
    script_path = Path("scripts", "check_portable_assets.py").resolve()
    spec = importlib.util.spec_from_file_location("check_portable_assets", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_model_tree(root: Path, name: str) -> None:
    model_dir = root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")


def test_build_summary_passes_with_all_required_assets(tmp_path: Path) -> None:
    module = _load_check_portable_assets()
    artifacts_root = tmp_path / "artifacts"
    hf_root = artifacts_root / "hf_models"
    local_vlm_root = artifacts_root / "local_vlm"

    for name in ("clip", "blip_itm", "vilt_itm", "bert_mlm"):
        _write_model_tree(hf_root, name)
    for name in LOCAL_OPENAI_COMPAT_LOCAL_DIRS:
        _write_model_tree(local_vlm_root, name)

    summary = module.build_summary(artifacts_root)

    assert summary["missing_hf_models"] == []
    assert summary["missing_local_vlm_models"] == []


def test_build_summary_reports_missing_local_vlm_assets(tmp_path: Path) -> None:
    module = _load_check_portable_assets()
    artifacts_root = tmp_path / "artifacts"
    hf_root = artifacts_root / "hf_models"
    local_vlm_root = artifacts_root / "local_vlm"

    for name in ("clip", "blip_itm", "vilt_itm", "bert_mlm"):
        _write_model_tree(hf_root, name)
    for name in [name for name in LOCAL_OPENAI_COMPAT_LOCAL_DIRS if name != "minicpm_v"]:
        _write_model_tree(local_vlm_root, name)

    summary = module.build_summary(artifacts_root)

    assert summary["missing_hf_models"] == []
    assert summary["missing_local_vlm_models"] == ["minicpm_v"]
