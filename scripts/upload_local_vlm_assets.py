#!/usr/bin/env python3
# 文件说明：该文件属于运维与实验脚本，集中实现 upload local vlm assets 相关逻辑。
from __future__ import annotations

import argparse
import os
import posixpath
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import paramiko


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_LOCAL_DIRS

DEFAULT_LOCAL_ROOT = PROJECT_ROOT / "artifacts" / "local_vlm"
DEFAULT_REMOTE_ROOT = os.getenv("MMSEC_UPLOAD_REMOTE_ROOT", "/HARD-DATA/bks/aat-runtime/app/artifacts/local_vlm")
DEFAULT_MODEL_NAMES = ",".join(LOCAL_OPENAI_COMPAT_LOCAL_DIRS)


# 中文注释：定义 UploadTarget 的结构化职责，作为运维与实验脚本中状态、配置或行为的边界。
@dataclass(frozen=True)
class UploadTarget:
    name: str
    local_dir: Path
    remote_dir: str


# 中文注释：实现 parse_args 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local offline VLM assets to the server.")
    parser.add_argument("--host", default=os.getenv("MMSEC_UPLOAD_HOST", ""))
    parser.add_argument("--port", type=int, default=int(os.getenv("MMSEC_UPLOAD_PORT", "22")))
    parser.add_argument("--username", default=os.getenv("MMSEC_UPLOAD_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("MMSEC_UPLOAD_PASSWORD", ""))
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--models",
        default=DEFAULT_MODEL_NAMES,
        help="Comma-separated local_vlm directory names.",
    )
    return parser.parse_args()


# 中文注释：实现 ensure_remote_dir 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = []
    current = remote_dir
    while current not in ("", "/"):
        parts.append(current)
        current = posixpath.dirname(current)
    for directory in reversed(parts):
        try:
            sftp.stat(directory)
        except FileNotFoundError:
            sftp.mkdir(directory)


# 中文注释：实现 remote_file_size 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def remote_file_size(sftp: paramiko.SFTPClient, remote_path: str) -> int | None:
    try:
        return sftp.stat(remote_path).st_size
    except FileNotFoundError:
        return None


# 中文注释：实现 iter_files 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if path.suffix in {".lock", ".metadata", ".incomplete"}:
            continue
        files.append(path)
    return sorted(files)


# 中文注释：实现 build_targets 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def build_targets(local_root: Path, remote_root: str, names: list[str]) -> list[UploadTarget]:
    targets: list[UploadTarget] = []
    for name in names:
        local_dir = local_root / name
        if not local_dir.exists():
            raise FileNotFoundError(f"Local model directory not found: {local_dir}")
        if not (local_dir / "config.json").exists():
            raise FileNotFoundError(f"Missing config.json in {local_dir}")
        targets.append(UploadTarget(name=name, local_dir=local_dir, remote_dir=posixpath.join(remote_root, name)))
    return targets


# 中文注释：实现 upload_tree 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def upload_tree(sftp: paramiko.SFTPClient, target: UploadTarget) -> tuple[int, int]:
    ensure_remote_dir(sftp, target.remote_dir)
    uploaded = 0
    skipped = 0
    files = iter_files(target.local_dir)
    print(f"[start] {target.name}: {len(files)} files", flush=True)
    for file_path in files:
        relative = file_path.relative_to(target.local_dir).as_posix()
        remote_path = posixpath.join(target.remote_dir, relative)
        ensure_remote_dir(sftp, posixpath.dirname(remote_path))
        local_size = file_path.stat().st_size
        existing_size = remote_file_size(sftp, remote_path)
        if existing_size == local_size:
            skipped += 1
            print(f"[skip] {target.name}/{relative} ({local_size} bytes)", flush=True)
            continue
        print(f"[upload] {target.name}/{relative} ({local_size} bytes)", flush=True)
        sftp.put(str(file_path), remote_path)
        uploaded += 1
    print(f"[done] {target.name}: uploaded={uploaded} skipped={skipped}", flush=True)
    return uploaded, skipped


# 中文注释：串联 main 的主流程，集中处理运维与实验脚本的初始化、执行和退出条件。
def main() -> int:
    args = parse_args()
    missing_connection = [name for name in ("host", "username", "password") if not str(getattr(args, name) or "").strip()]
    if missing_connection:
        raise SystemExit("Missing upload connection option(s): " + ", ".join(missing_connection))
    local_root = Path(args.local_root).resolve()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    targets = build_targets(local_root, args.remote_root, model_names)

    transport = paramiko.Transport((args.host, args.port))
    transport.connect(username=args.username, password=args.password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        total_uploaded = 0
        total_skipped = 0
        for target in targets:
            uploaded, skipped = upload_tree(sftp, target)
            total_uploaded += uploaded
            total_skipped += skipped
        print(f"[summary] uploaded={total_uploaded} skipped={total_skipped}", flush=True)
    finally:
        sftp.close()
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
