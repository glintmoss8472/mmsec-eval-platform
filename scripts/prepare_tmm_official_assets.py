from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


UTC = timezone.utc
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class UrlAsset:
    name: str
    url: str
    dest: str
    aliases: tuple[str, ...] = ()
    required_for_audit: bool = True


@dataclass(frozen=True)
class DriveAsset:
    name: str
    drive_id: str
    dest: str
    is_folder: bool = False
    required_for_audit: bool = True


URL_ASSETS = [
    UrlAsset(
        "albef_downstream_json",
        "https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz",
        "downloads/albef/data.tar.gz",
        required_for_audit=True,
    ),
    UrlAsset(
        "albef_mscoco_retrieval",
        "https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/mscoco.pth",
        "checkpoints/ALBEF/mscoco.pth",
    ),
    UrlAsset(
        "albef_flickr30k_retrieval",
        "https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth",
        "checkpoints/ALBEF/flickr30k.pth",
    ),
    UrlAsset(
        "albef_refcoco_grounding",
        "https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/refcoco.pth",
        "checkpoints/ALBEF/refcoco.pth",
        required_for_audit=False,
    ),
    UrlAsset(
        "vilt_mscoco_retrieval",
        "https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_coco.ckpt",
        "checkpoints/ViLT/mscoco.ckpt",
    ),
    UrlAsset(
        "vilt_flickr30k_retrieval",
        "https://github.com/dandelin/ViLT/releases/download/200k/vilt_irtr_f30k.ckpt",
        "checkpoints/ViLT/flickr30k.ckpt",
    ),
    UrlAsset(
        "meter_mscoco_retrieval",
        "https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_coco.ckpt",
        "checkpoints/METER/mscoco.ckpt",
    ),
    UrlAsset(
        "meter_flickr30k_retrieval",
        "https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_flickr.ckpt",
        "checkpoints/METER/flickr30k.ckpt",
    ),
    UrlAsset(
        "meter_snli_ve",
        "https://github.com/zdou0830/METER/releases/download/checkpoint/meter_clip16_288_roberta_snli.ckpt",
        "checkpoints/METER/snli-ve.ckpt",
        required_for_audit=False,
    ),
    UrlAsset(
        "blip_mscoco_retrieval",
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
        "checkpoints/BLIP/mscoco.pth",
    ),
    UrlAsset(
        "blip_flickr30k_retrieval",
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth",
        "checkpoints/BLIP/flickr30k.pth",
    ),
    UrlAsset(
        "tcl_mscoco_retrieval_hf",
        "https://huggingface.co/Sensen02/VLPTransferAttackCheckpoints/resolve/main/tcl_mscoco.pth?download=true",
        "checkpoints/TCL/mscoco.pth",
    ),
    UrlAsset(
        "tcl_flickr30k_retrieval_hf",
        "https://huggingface.co/Sensen02/VLPTransferAttackCheckpoints/resolve/main/tcl_flickr.pth?download=true",
        "checkpoints/TCL/flickr30k.pth",
    ),
]


FORBIDDEN_FOREIGN_HOST_SUFFIXES = (
    "github.com",
    "githubusercontent.com",
    "google.com",
    "googleapis.com",
    "googleusercontent.com",
    "gstatic.com",
    "huggingface.co",
    "hf.co",
    "xethub.hf.co",
)


DOMESTIC_RESOURCE_PROBES = [
    {
        "name": "flickr30k_atyun_listing",
        "url": "https://www.atyun.com/datasets/files/nlphuji/flickr30k.html",
        "expected_text": "flickr30k-images.zip",
        "blocker_hint": "ATYUN lists the 4.09GB Flickr30K image archive, but file download may require a logged-in ATYUN session.",
    },
    {
        "name": "flickr30k_gitee_ai_listing",
        "url": "https://ai.gitee.com/hf-datasets/HuggingFaceM4/flickr30k",
        "expected_text": "flickr30k",
        "blocker_hint": "Gitee AI exposes a domestic dataset page, but the Git/LFS endpoint may require an authenticated Gitee account.",
    },
    {
        "name": "flickr30k_openxlab_listing",
        "url": "https://openxlab.org.cn/datasets/OpenDataLab/Flickr_Image",
        "expected_text": "Flickr_Image",
        "blocker_hint": "OpenXLab/OpenDataLab lists an 8.16GB Flickr30k archive.zip under OpenDataLab/Flickr_Image, but even small file downloads require openxlab login AK/SK.",
    },
]


DRIVE_ASSETS = [
    DriveAsset(
        "tcl_retrieval_checkpoints_zip",
        "1eHinvFP7TnZYAL2Ft-M8rPott7mpVN2R",
        "downloads/tcl/TCL_ckpt.zip",
    ),
    DriveAsset(
        "xvlm_4m_all_finetuned_checkpoints",
        "1laNJHBnVGF7onbEYh1vO-b2P5TxdqH-k",
        "downloads/xvlm/4m_base_finetune.tar",
    ),
    DriveAsset(
        "xvlm_16m_itr_coco_folder",
        "1VotCNmdevvtMuJmdxPfg3MOZXJRnV96D",
        "downloads/xvlm/itr_coco",
        is_folder=True,
        required_for_audit=False,
    ),
    DriveAsset(
        "xvlm_16m_itr_flickr_folder",
        "1lsuBVP7MEqGqWkqRxaxb8N8TbSKqQ1Yz",
        "downloads/xvlm/itr_flickr",
        is_folder=True,
        required_for_audit=False,
    ),
    DriveAsset(
        "xvlm_16m_refcoco_folder",
        "1ySQTjpTm5CeHp50YYFObUjT7DTHLN7DZ",
        "downloads/xvlm/refcoco_bbox",
        is_folder=True,
        required_for_audit=False,
    ),
]


EXTERNAL_REPOS = {
    "whdii_TMM": "https://github.com/whdii/TMM.git",
    "ALBEF": "https://github.com/salesforce/ALBEF.git",
    "TCL": "https://github.com/uta-smile/TCL.git",
    "X-VLM": "https://github.com/zengyan-97/X-VLM.git",
    "ViLT": "https://github.com/dandelin/ViLT.git",
    "METER": "https://github.com/zdou0830/METER.git",
}


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else "",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _host_matches(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith(f".{suffix}")


def _is_forbidden_foreign_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(_host_matches(host, suffix) for suffix in FORBIDDEN_FOREIGN_HOST_SUFFIXES)


def _download_url(url: str, dest: Path, *, timeout: int, retries: int, domestic_only: bool = False) -> dict[str, Any]:
    if domestic_only and _is_forbidden_foreign_url(url):
        return {"status": "blocked_foreign_url", "path": str(dest), "url": url}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size > 0:
        return {"status": "exists", "path": str(dest), "size_bytes": dest.stat().st_size}
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp.replace(dest)
            return {"status": "downloaded", "path": str(dest), "size_bytes": dest.stat().st_size, "attempt": attempt}
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = repr(exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    return {"status": "failed", "path": str(dest), "url": url, "error": last_error}


def _probe_domestic_sources(*, timeout: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for probe in DOMESTIC_RESOURCE_PROBES:
        url = str(probe["url"])
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read(2_000_000).decode("utf-8", "ignore")
            expected = str(probe["expected_text"]).lower()
            lower_body = body.lower()
            status = "listed" if expected in lower_body else "page_reachable"
            if "login" in lower_body or "member/login" in lower_body:
                status = "login_required"
            results.append(
                {
                    "name": probe["name"],
                    "status": status,
                    "url": url,
                    "contains_expected_text": expected in lower_body,
                    "blocker_hint": probe["blocker_hint"],
                }
            )
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            results.append({"name": probe["name"], "status": "failed", "url": url, "error": repr(exc)})
    return results


def _prepare_coco_val2014_from_autodl(coco2017_root: Path, tmm_root: Path, *, limit: int = 0) -> dict[str, Any]:
    test_json = tmm_root / "datasets" / "coco_test.json"
    output_dir = tmm_root / "datasets" / "mscoco" / "val2014"
    archives = [coco2017_root / "val2017.zip", coco2017_root / "train2017.zip"]
    if not test_json.exists():
        return {"status": "missing_coco_test_json", "path": str(test_json)}
    missing_archives = [str(path) for path in archives if not path.exists()]
    if missing_archives:
        return {"status": "missing_autodl_coco_archives", "missing": missing_archives}

    samples = json.loads(test_json.read_text(encoding="utf-8"))
    image_names: list[str] = []
    for sample in samples:
        rel = str(sample.get("image", ""))
        name = Path(rel).name
        if name.startswith("COCO_val2014_") and name.endswith(".jpg"):
            image_names.append(name)
    if limit > 0:
        image_names = image_names[:limit]
    expected = dict.fromkeys(image_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = {name for name in expected if not (output_dir / name).exists()}
    if not pending:
        return {"status": "exists", "expected": len(expected), "found": len(expected), "output_dir": str(output_dir)}

    found = len(expected) - len(pending)
    by_coco2017_name = {f"{name[-16:-4]}.jpg": name for name in pending}
    extracted_by_archive: dict[str, int] = {}
    for archive in archives:
        archive_count = 0
        prefix = archive.stem + "/"
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                short_name = info.filename.removeprefix(prefix)
                target_name = by_coco2017_name.get(short_name)
                if not target_name:
                    continue
                with zf.open(info) as src, (output_dir / target_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                archive_count += 1
                found += 1
                pending.discard(target_name)
                by_coco2017_name.pop(short_name, None)
                if not pending:
                    break
        extracted_by_archive[archive.name] = archive_count
        if not pending:
            break

    report = {
        "status": "prepared" if not pending else "incomplete",
        "expected": len(expected),
        "found": found,
        "missing": len(pending),
        "output_dir": str(output_dir),
        "source_root": str(coco2017_root),
        "extracted_by_archive": extracted_by_archive,
    }
    _write_json(tmm_root / "datasets" / "mscoco" / "COCO2017_EXTRACTION_REPORT.json", report)
    return report


def _ensure_gdown() -> dict[str, Any]:
    proc = subprocess.run([sys.executable, "-m", "gdown", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        return {"status": "present", "stdout": proc.stdout.strip()}
    install = _run([sys.executable, "-m", "pip", "install", "-q", "gdown"], timeout=300)
    if install["returncode"] != 0:
        return {"status": "failed", "install": install}
    proc2 = subprocess.run([sys.executable, "-m", "gdown", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"status": "present" if proc2.returncode == 0 else "failed", "stdout": proc2.stdout.strip(), "stderr": proc2.stderr.strip()}


def _download_drive(asset: DriveAsset, asset_root: Path, *, timeout: int) -> dict[str, Any]:
    dest = asset_root / asset.dest
    if dest.exists() and (dest.is_dir() or dest.stat().st_size > 0):
        size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) if dest.is_dir() else dest.stat().st_size
        if size > 0:
            return {"name": asset.name, "status": "exists", "path": str(dest), "size_bytes": size}
    gdown_status = _ensure_gdown()
    if gdown_status.get("status") != "present":
        return {"name": asset.name, "status": "failed", "path": str(dest), "gdown": gdown_status}
    dest.parent.mkdir(parents=True, exist_ok=True)
    if asset.is_folder:
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "gdown", "--folder", "--remaining-ok", asset.drive_id, "-O", str(dest)]
    else:
        cmd = [sys.executable, "-m", "gdown", asset.drive_id, "-O", str(dest)]
    result = _run(cmd, timeout=timeout)
    status = "downloaded" if result["returncode"] == 0 and dest.exists() else "failed"
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) if dest.exists() and dest.is_dir() else (dest.stat().st_size if dest.exists() and dest.is_file() else 0)
    return {"name": asset.name, "status": status, "path": str(dest), "size_bytes": size, "result": result}


def _clone_repo(name: str, url: str, repo_root: Path) -> dict[str, Any]:
    dest = repo_root / name
    if (dest / ".git").exists():
        result = _run(["git", "-C", str(dest), "pull", "--ff-only"], timeout=300)
        return {"name": name, "status": "updated" if result["returncode"] == 0 else "failed", "path": str(dest), "result": result}
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = _run(["git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1", url, str(dest)], timeout=600)
    return {"name": name, "status": "cloned" if result["returncode"] == 0 else "failed", "path": str(dest), "result": result}


def _safe_symlink_or_copy(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        return {"status": "missing_source", "src": str(src), "dst": str(dst)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return {"status": "exists", "src": str(src), "dst": str(dst)}
    try:
        os.symlink(src, dst)
        return {"status": "symlinked", "src": str(src), "dst": str(dst)}
    except OSError:
        shutil.copy2(src, dst)
        return {"status": "copied", "src": str(src), "dst": str(dst)}


def _extract_archives(asset_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    data_tar = asset_root / "downloads" / "albef" / "data.tar.gz"
    if data_tar.exists():
        out_dir = asset_root / "downloads" / "albef" / "data"
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(data_tar, "r:gz") as tar:
                    tar.extractall(out_dir)
                results.append({"name": "albef_downstream_json", "status": "extracted", "path": str(out_dir)})
            except (OSError, tarfile.TarError) as exc:
                results.append({"name": "albef_downstream_json", "status": "failed", "error": repr(exc)})
        else:
            results.append({"name": "albef_downstream_json", "status": "exists", "path": str(out_dir)})

    tcl_zip = asset_root / "downloads" / "tcl" / "TCL_ckpt.zip"
    if tcl_zip.exists():
        out_dir = asset_root / "downloads" / "tcl" / "TCL_ckpt"
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(tcl_zip) as zf:
                    zf.extractall(out_dir)
                results.append({"name": "tcl_retrieval_checkpoints_zip", "status": "extracted", "path": str(out_dir)})
            except (OSError, zipfile.BadZipFile) as exc:
                results.append({"name": "tcl_retrieval_checkpoints_zip", "status": "failed", "error": repr(exc)})
        else:
            results.append({"name": "tcl_retrieval_checkpoints_zip", "status": "exists", "path": str(out_dir)})

    xvlm_tar = asset_root / "downloads" / "xvlm" / "4m_base_finetune.tar"
    if xvlm_tar.exists():
        out_dir = asset_root / "downloads" / "xvlm" / "4m_base_finetune"
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(xvlm_tar) as tar:
                    tar.extractall(out_dir)
                results.append({"name": "xvlm_4m_all_finetuned_checkpoints", "status": "extracted", "path": str(out_dir)})
            except (OSError, tarfile.TarError) as exc:
                results.append({"name": "xvlm_4m_all_finetuned_checkpoints", "status": "failed", "error": repr(exc)})
        else:
            results.append({"name": "xvlm_4m_all_finetuned_checkpoints", "status": "exists", "path": str(out_dir)})
    return results


def _find_first(root: Path, patterns: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    lower_patterns = tuple(p.lower() for p in patterns)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        full = str(path).lower()
        if all(p in full or p in name for p in lower_patterns):
            return path
    return None


def _link_assets(asset_root: Path, tmm_root: Path) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for rel in (
        "checkpoints/ALBEF/mscoco.pth",
        "checkpoints/ALBEF/flickr30k.pth",
        "checkpoints/ViLT/mscoco.ckpt",
        "checkpoints/ViLT/flickr30k.ckpt",
        "checkpoints/METER/mscoco.ckpt",
        "checkpoints/METER/flickr30k.ckpt",
        "checkpoints/BLIP/mscoco.pth",
        "checkpoints/BLIP/flickr30k.pth",
        "checkpoints/TCL/mscoco.pth",
        "checkpoints/TCL/flickr30k.pth",
        "checkpoints/X-VLM/mscoco.pth",
        "checkpoints/X-VLM/flickr30k.pth",
    ):
        links.append(_safe_symlink_or_copy(asset_root / rel, tmm_root / rel))

    tcl_root = asset_root / "downloads" / "tcl" / "TCL_ckpt"
    tcl_coco = _find_first(tcl_root, ("coco",))
    tcl_flickr = _find_first(tcl_root, ("flickr",))
    if tcl_coco:
        links.append(_safe_symlink_or_copy(tcl_coco, tmm_root / "checkpoints" / "TCL" / "mscoco.pth"))
    if tcl_flickr:
        links.append(_safe_symlink_or_copy(tcl_flickr, tmm_root / "checkpoints" / "TCL" / "flickr30k.pth"))

    xvlm_root = asset_root / "downloads" / "xvlm"
    xvlm_coco = _find_first(xvlm_root, ("coco",))
    xvlm_flickr = _find_first(xvlm_root, ("flickr",))
    if xvlm_coco:
        links.append(_safe_symlink_or_copy(xvlm_coco, tmm_root / "checkpoints" / "X-VLM" / "mscoco.pth"))
    if xvlm_flickr:
        links.append(_safe_symlink_or_copy(xvlm_flickr, tmm_root / "checkpoints" / "X-VLM" / "flickr30k.pth"))

    albef_data_root = asset_root / "downloads" / "albef" / "data"
    for src_name, dst_name in (
        ("flickr30k_test.json", "flickr30k_test.json"),
        ("coco_test.json", "coco_test.json"),
        ("refcoco+_test.json", "refcoco+_test.json"),
        ("refcoco+_val.json", "refcoco+_val.json"),
        ("snli_ve_test.json", "snli_ve_test.json"),
        ("ve_test.json", "snli_ve_test.json"),
    ):
        src = _find_first(albef_data_root, (src_name.lower(),))
        if src:
            links.append(_safe_symlink_or_copy(src, tmm_root / "datasets" / dst_name))
    return links


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare official assets referenced by whdii/TMM and linked upstream VLP repositories.")
    parser.add_argument("--asset-root", default="/root/autodl-tmp/paper_assets/tmm_official")
    parser.add_argument("--repo-root", default="/root/autodl-tmp/paper_repos")
    parser.add_argument("--tmm-root", default="/root/autodl-tmp/paper_repos/TMM-main/TMM-main")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--skip-url-downloads", action="store_true")
    parser.add_argument("--domestic-only", action="store_true", help="Do not access GitHub, Google, Hugging Face, Google Drive, or other known foreign asset hosts.")
    parser.add_argument("--prepare-coco-from-autodl", action="store_true", help="Build the TMM COCO val2014 image folder from AutoDL public COCO2017 archives.")
    parser.add_argument("--autodl-coco2017-root", default="/autodl-pub/data/COCO2017")
    parser.add_argument("--coco-extract-limit", type=int, default=0)
    parser.add_argument("--include-google-drive", action="store_true")
    parser.add_argument("--clone-repos", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    asset_root = Path(args.asset_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    tmm_root = Path(args.tmm_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else PROJECT_ROOT / "artifacts" / f"tmm_asset_prepare_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    url_results = []
    if not args.skip_url_downloads:
        for asset in URL_ASSETS:
            url_results.append({"name": asset.name, **_download_url(asset.url, asset_root / asset.dest, timeout=args.timeout, retries=args.retries, domestic_only=args.domestic_only)})

    drive_results = []
    if args.include_google_drive and not args.domestic_only:
        for asset in DRIVE_ASSETS:
            drive_results.append(_download_drive(asset, asset_root, timeout=args.timeout))
    elif args.include_google_drive and args.domestic_only:
        drive_results.append({"name": "google_drive_assets", "status": "blocked_foreign_url", "reason": "--domestic-only"})

    repo_results = []
    if args.clone_repos and not args.domestic_only:
        for name, url in EXTERNAL_REPOS.items():
            repo_results.append(_clone_repo(name, url, repo_root))
    elif args.clone_repos and args.domestic_only:
        repo_results.append({"name": "external_repos", "status": "blocked_foreign_url", "reason": "--domestic-only"})

    domestic_results: dict[str, Any] = {}
    if args.domestic_only:
        domestic_results["source_probes"] = _probe_domestic_sources(timeout=min(args.timeout, 30))
    if args.domestic_only or args.prepare_coco_from_autodl:
        domestic_results["coco_autodl"] = _prepare_coco_val2014_from_autodl(
            Path(args.autodl_coco2017_root).resolve(),
            tmm_root,
            limit=args.coco_extract_limit,
        )

    extract_results = _extract_archives(asset_root)
    link_results = _link_assets(asset_root, tmm_root)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "asset_root": str(asset_root),
        "repo_root": str(repo_root),
        "tmm_root": str(tmm_root),
        "url_results": url_results,
        "drive_results": drive_results,
        "repo_results": repo_results,
        "domestic_results": domestic_results,
        "extract_results": extract_results,
        "link_results": link_results,
    }
    _write_json(out_dir / "tmm_official_asset_prepare.json", payload)
    print(json.dumps({"out_dir": str(out_dir), "url": len(url_results), "drive": len(drive_results), "repos": len(repo_results)}, ensure_ascii=False))
    failed = [x for group in (url_results, drive_results, repo_results, extract_results, link_results) for x in group if str(x.get("status")) in {"failed", "missing_source"} and x.get("required_for_audit", True)]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
