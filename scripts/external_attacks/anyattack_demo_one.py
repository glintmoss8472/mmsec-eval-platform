# 文件说明：该文件属于外部攻击脚本，集中实现 anyattack demo one 相关逻辑。
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# 中文注释：串联 main 的主流程，集中处理外部攻击脚本的初始化、执行和退出条件。
def main() -> int:
    parser = argparse.ArgumentParser(description="Run official AnyAttack demo.py for one image pair.")
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--output_image", required=True)
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--target_image", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--python_bin", default=sys.executable)
    args = parser.parse_args()

    repo = Path(args.repo_dir).expanduser().resolve()
    demo = repo / "demo.py"
    if not demo.exists():
        raise FileNotFoundError(f"AnyAttack demo.py not found: {demo}")
    decoder = Path(args.decoder_path).expanduser().resolve()
    if not decoder.exists():
        raise FileNotFoundError(f"AnyAttack decoder_path not found: {decoder}")
    target = Path(args.target_image).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"AnyAttack target_image not found: {target}")
    output = Path(args.output_image).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        args.python_bin,
        str(demo),
        "--decoder_path", str(decoder),
        "--clean_image_path", str(Path(args.input_image).resolve()),
        "--target_image_path", str(target),
        "--output_path", str(output),
        "--device", args.device,
        "--eps", str(float(args.epsilon)),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo))
    proc = subprocess.run(command, cwd=str(repo), env=env)
    if proc.returncode != 0:
        return proc.returncode
    if not output.exists():
        raise FileNotFoundError(f"AnyAttack did not create expected output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
