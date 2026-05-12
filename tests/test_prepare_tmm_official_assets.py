# 文件说明：该文件属于自动化测试，集中实现 test prepare tmm official assets 相关逻辑。
from __future__ import annotations

from pathlib import Path
import json
import zipfile

from scripts.prepare_tmm_official_assets import (
    _is_forbidden_foreign_url,
    _prepare_coco_val2014_from_autodl,
    _safe_symlink_or_copy,
)


# 中文注释：验证 test_safe_symlink_or_copy_links_existing_file 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_safe_symlink_or_copy_links_existing_file(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    dst = tmp_path / "nested" / "dst.bin"
    src.write_bytes(b"checkpoint")

    result = _safe_symlink_or_copy(src, dst)

    assert result["status"] in {"symlinked", "copied"}
    assert dst.exists()
    assert dst.read_bytes() == b"checkpoint"


# 中文注释：验证 test_safe_symlink_or_copy_reports_missing_source 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_safe_symlink_or_copy_reports_missing_source(tmp_path: Path) -> None:
    result = _safe_symlink_or_copy(tmp_path / "missing.bin", tmp_path / "dst.bin")

    assert result["status"] == "missing_source"


# 中文注释：验证 test_domestic_only_blocks_known_foreign_asset_hosts 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_domestic_only_blocks_known_foreign_asset_hosts() -> None:
    assert _is_forbidden_foreign_url("https://github.com/whdii/TMM/releases/download/a/b.pth")
    assert _is_forbidden_foreign_url("https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/mscoco.pth")
    assert _is_forbidden_foreign_url("https://huggingface.co/example/model/resolve/main/file.bin")
    assert not _is_forbidden_foreign_url("https://www.atyun.com/datasets/files/nlphuji/flickr30k.html")
    assert not _is_forbidden_foreign_url("https://ai.gitee.com/hf-datasets/HuggingFaceM4/flickr30k")


# 中文注释：验证 test_prepare_coco_val2014_from_autodl_extracts_named_images 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_prepare_coco_val2014_from_autodl_extracts_named_images(tmp_path: Path) -> None:
    tmm_root = tmp_path / "TMM-main"
    dataset_dir = tmm_root / "datasets"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "coco_test.json").write_text(
        json.dumps([{"image": "val2014/COCO_val2014_000000391895.jpg", "caption": ["caption"]}]),
        encoding="utf-8",
    )
    coco_root = tmp_path / "COCO2017"
    coco_root.mkdir()
    with zipfile.ZipFile(coco_root / "val2017.zip", "w") as zf:
        zf.writestr("val2017/000000391895.jpg", b"image-bytes")
    with zipfile.ZipFile(coco_root / "train2017.zip", "w") as zf:
        zf.writestr("train2017/000000000000.jpg", b"unused")

    result = _prepare_coco_val2014_from_autodl(coco_root, tmm_root)

    assert result["status"] == "prepared"
    out = dataset_dir / "mscoco" / "val2014" / "COCO_val2014_000000391895.jpg"
    assert out.read_bytes() == b"image-bytes"
