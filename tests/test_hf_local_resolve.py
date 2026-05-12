from __future__ import annotations

from pathlib import Path

from mmsec_eval.model_adapters.hf_local import resolve_hf_model_source


def test_resolve_hf_model_source_prefers_local_dir_when_ready(tmp_path: Path, monkeypatch):
    art = tmp_path / "artifacts"
    model_dir = art / "hf_models" / "blip_itm"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "pytorch_model.bin").write_bytes(b"x")
    monkeypatch.setenv("MMSEC_ARTIFACTS_DIR", str(art))

    src = resolve_hf_model_source("Salesforce/blip-itm-base-coco", local_only=True, local_dir_name="blip_itm")
    assert Path(src).resolve() == model_dir.resolve()


def test_resolve_hf_model_source_keeps_repo_id_when_local_missing(tmp_path: Path, monkeypatch):
    art = tmp_path / "artifacts"
    monkeypatch.setenv("MMSEC_ARTIFACTS_DIR", str(art))
    monkeypatch.delenv("MMSEC_BUNDLE_ROOT", raising=False)
    monkeypatch.setenv("MMSEC_PROJECT_ARTIFACTS_FALLBACK", "0")

    src = resolve_hf_model_source("dandelin/vilt-b32-finetuned-coco", local_only=True, local_dir_name="vilt_itm")
    assert src == "dandelin/vilt-b32-finetuned-coco"


def test_resolve_hf_model_source_falls_back_to_bundle_assets(tmp_path: Path, monkeypatch):
    art = tmp_path / "artifacts"
    bundle_root = tmp_path / "bundle"
    model_dir = bundle_root / "artifacts" / "hf_models" / "clip"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("MMSEC_ARTIFACTS_DIR", str(art))
    monkeypatch.setenv("MMSEC_BUNDLE_ROOT", str(bundle_root))

    src = resolve_hf_model_source("openai/clip-vit-base-patch32", local_only=True, local_dir_name="clip")
    assert Path(src).resolve() == model_dir.resolve()
