# 文件说明：该文件属于自动化测试，集中实现 test prepare scripts smoke 相关逻辑。
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import importlib.util


# 中文注释：封装 _run_py 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _run_py(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, script] + args
    return subprocess.run(cmd, capture_output=True, text=True)


# 中文注释：封装 _run_ps 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _run_ps(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script] + args
    return subprocess.run(cmd, capture_output=True, text=True)


# 中文注释：验证 test_prepare_flickr30k_script_smoke 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_prepare_flickr30k_script_smoke(tmp_path: Path):
    root = tmp_path / "flickr_data"
    img = root / "images"
    img.mkdir(parents=True, exist_ok=True)

    arr = np.zeros((24, 24, 3), dtype=np.uint8)
    arr[..., 0] = 200
    Image.fromarray(arr).save(img / "x.png")
    (root / "results_20130124.token").write_text("x.png#0\ta sample caption\n", encoding="utf-8")

    res = _run_ps("scripts/prepare_flickr30k.ps1", ["-Root", str(root), "-ImageDir", "images"])
    assert res.returncode == 0, res.stderr + "\n" + res.stdout
    assert (root / "captions_index.jsonl").exists()


# 中文注释：验证 test_prepare_coco_subset_script_smoke 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_prepare_coco_subset_script_smoke(tmp_path: Path):
    root = tmp_path / "coco_data"
    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    img = root / "val2017"
    img.mkdir(parents=True, exist_ok=True)

    # Ensure the script won't attempt to download images (it checks whether img_dir has any files).
    arr = np.zeros((24, 24, 3), dtype=np.uint8)
    arr[..., 1] = 200
    Image.fromarray(arr).save(img / "a.jpg")

    payload = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [{"id": 10, "image_id": 1, "caption": "hello"}],
    }
    (ann / "captions_val2017.json").write_text(json.dumps(payload), encoding="utf-8")

    res = _run_ps(
        "scripts/prepare_coco_subset.ps1",
        ["-Root", str(root), "-Split", "val2017", "-MaxItems", "1"],
    )
    assert res.returncode == 0, res.stderr + "\n" + res.stdout
    assert (ann / "captions_val2017_subset.json").exists()


# 中文注释：验证 test_prepare_flickr30k_python_script_rejects_placeholder_fallback 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_prepare_flickr30k_python_script_rejects_placeholder_fallback(tmp_path: Path):
    root = tmp_path / "flickr_placeholder"
    img = root / "images"
    img.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(img / "demo_0000.jpg")
    (root / "captions.jsonl").write_text(
        json.dumps(
            {
                "id": "demo-0000",
                "image": "demo_0000.jpg",
                "caption": "Demo caption 0000: multimodal retrieval placeholder sample.",
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    res = _run_py(
        "scripts/prepare_flickr30k.py",
        ["-Root", str(root), "-CaptionsSource", "captions.jsonl", "-AutoDownload", "false"],
    )
    assert res.returncode != 0, res.stderr + "\n" + res.stdout
    assert "Synthetic fallback is disabled by default" in res.stdout


# 中文注释：验证 test_prepare_flickr1k_python_script_uses_tmm_official_assets 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_prepare_flickr1k_python_script_uses_tmm_official_assets(tmp_path: Path):
    source = tmp_path / "tmm_datasets"
    image_dir = source / "flickr" / "flickr30k-images"
    image_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(image_dir / "a.jpg")
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(image_dir / "b.jpg")
    (source / "flickr30k_test.json").write_text(
        json.dumps(
            [
                {"image": "flickr30k-images/a.jpg", "caption": ["alpha caption", "second caption"]},
                {"image": "flickr30k-images/b.jpg", "caption": ["beta caption"]},
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "flickr1k"

    res = _run_py(
        "scripts/prepare_flickr1k.py",
        [
            "-Root",
            str(out),
            "-SourceRoot",
            str(source),
            "-OutputFile",
            "captions_index_single.jsonl",
            "-AutoDownload",
            "false",
            "-MaxItems",
            "1",
        ],
    )

    assert res.returncode == 0, res.stderr + "\n" + res.stdout
    rows = [json.loads(line) for line in (out / "captions_index_single.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "flickr1k-0000", "image": "images/a.jpg", "caption": "alpha caption", "split": "test"}]
    assert (out / "images" / "a.jpg").exists()


# 中文注释：验证 test_prepare_coco_subset_python_script_uses_real_local_images 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_prepare_coco_subset_python_script_uses_real_local_images(tmp_path: Path):
    root = tmp_path / "coco_real"
    ann = root / "annotations"
    img = root / "val2017"
    ann.mkdir(parents=True, exist_ok=True)
    img.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(img / "000000000001.jpg")
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(img / "000000000002.jpg")
    payload = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 24, "height": 24},
            {"id": 2, "file_name": "000000000002.jpg", "width": 24, "height": 24},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "caption": "real sample one"},
            {"id": 11, "image_id": 2, "caption": "real sample two"},
        ],
    }
    (ann / "captions_val2017.json").write_text(json.dumps(payload), encoding="utf-8")
    res = _run_py(
        "scripts/prepare_coco_subset.py",
        ["-Root", str(root), "-Split", "val2017", "-MaxItems", "2", "-AutoDownload", "false"],
    )
    assert res.returncode == 0, res.stderr + "\n" + res.stdout
    assert (ann / "captions_val2017_subset.json").exists()
    assert (ann / "captions_val2017_subset.jsonl").exists()


# 中文注释：验证 test_run_thesis_matrix_help_imports_current_local_vlm_catalog 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_run_thesis_matrix_help_imports_current_local_vlm_catalog():
    res = _run_py("scripts/run_thesis_matrix.py", ["--help"])

    assert res.returncode == 0, res.stderr + "\n" + res.stdout
    assert "--entry-ids" in res.stdout


# 中文注释：验证 test_prepare_flickr30k_can_extract_real_rows_from_release_archive 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_prepare_flickr30k_can_extract_real_rows_from_release_archive(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("prepare_flickr30k", "scripts/prepare_flickr30k.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset_root = tmp_path / "flickr_release"
    image_dir = dataset_root / "images"
    cache_dir = dataset_root / ".download_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "flickr30k.zip"

    img = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8))
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(
            "flickr30k/captions.txt",
            "image,caption\n"
            "a.jpg,alpha caption\n"
            "a.jpg,beta caption\n"
            "b.jpg,gamma caption\n",
        )
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        zf.writestr("flickr30k/images/a.jpg", buf.getvalue())
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        zf.writestr("flickr30k/images/b.jpg", buf.getvalue())

    module._download_flickr30k_release = lambda root: (archive_path, cache_dir)  # type: ignore[attr-defined]
    rows = module._extract_flickr30k_release(dataset_root, image_dir, limit=1)  # type: ignore[attr-defined]
    assert len(rows) == 2
    assert {row["image"] for row in rows} == {"a.jpg"}
    assert (image_dir / "a.jpg").exists()
