# 文件说明：该文件属于外部攻击脚本，集中实现 vqa visual corruption one 相关逻辑。
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


OFFICIAL_ALIAS: dict[str, str] = {
    "shot_noise": "Shot-noise",
    "shot-noise": "Shot-noise",
    "gaussian_noise": "Gaussian-noise",
    "gaussian-noise": "Gaussian-noise",
    "impulse_noise": "Impulse-noise",
    "impulse-noise": "Impulse-noise",
    "speckle_noise": "Speckle-noise",
    "speckle-noise": "Speckle-noise",
    "gaussian_blur": "Defocus-blur",
    "defocus_blur": "Defocus-blur",
    "defocus-blur": "Defocus-blur",
    "motion_blur": "Zoom-Blur",
    "zoom_blur": "Zoom-Blur",
    "zoom-blur": "Zoom-Blur",
    "snow": "Snow",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "saturation": "Saturation",
    "saturate": "Saturation",
    "elastic": "Elastic",
    "resize_compress": "Pixelate",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG-compression",
    "jpeg-compression": "JPEG-compression",
    "occlusion": "Spatter",
    "spatter": "Spatter",
    "grayscale": "Grayscale",
    "grayscale_inverse": "Grayscale-Inverse",
    "grayscale-inverse": "Grayscale-Inverse",
}


# 实现 `_Logger.__init__` 的对象行为，维护该类在外部攻击脚本中的调用契约。
class _Logger:
    # 封装 _Logger.__init__ 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # 实现 `_Logger._write` 的对象行为，维护该类在外部攻击脚本中的调用契约。
    def _write(self, level: str, message: object) -> None:
        with (self.log_dir / "vqa_visual_corruption_official.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{level}: {message}\n")

    # 整理 `info`，描述当前服务器运行环境、模型入口或部署状态。
    def info(self, message: object) -> None:
        self._write("INFO", message)

    # 实现 `_Logger.warning` 的对象行为，维护该类在外部攻击脚本中的调用契约。
    def warning(self, message: object) -> None:
        self._write("WARNING", message)

    # 实现 `_Logger.error` 的对象行为，维护该类在外部攻击脚本中的调用契约。
    def error(self, message: object) -> None:
        self._write("ERROR", message)


# 实现 `_SingleImageDataset.__init__` 的对象行为，维护该类在外部攻击脚本中的调用契约。
class _SingleImageDataset:
    # 封装 _SingleImageDataset.__init__ 的内部步骤，让外部攻击脚本主流程保持清晰并隔离边界细节。
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path

    # 实现 `_SingleImageDataset.__len__` 的对象行为，维护该类在外部攻击脚本中的调用契约。
    def __len__(self) -> int:
        return 1

    # 实现 `_SingleImageDataset.__getitem__` 的对象行为，维护该类在外部攻击脚本中的调用契约。
    def __getitem__(self, idx: int):
        if idx != 0:
            raise IndexError(idx)
        image = np.asarray(Image.open(self.image_path).convert("RGB"))
        return image, [], [], 0, [], []


# 执行 `official key` 辅助逻辑，保持外部攻击脚本中的输入处理和结果输出一致。
def _official_key(corruption_type: str, severity: int) -> tuple[str, dict[str, Any]]:
    normalized = str(corruption_type or "").strip()
    if not normalized:
        normalized = "Gaussian-noise"
    key = normalized.replace(" ", "_").lower()
    base = OFFICIAL_ALIAS.get(key, normalized)
    if base in {"Grayscale", "Grayscale-Inverse"}:
        return base, {"official_corruption": base, "level": None}
    if "_L" in base:
        official_key = base
    else:
        official_key = f"{base}_L{severity}"
    return official_key, {"official_corruption": base, "level": severity}


# 写出 `图像`，保证后续报告、页面或复现实验能读取。
def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    Image.fromarray(arr[:, :, :3], mode="RGB").save(path)


# 作为 `vqa_visual_corruption_one.py` 的执行入口，串联参数读取、核心处理和退出状态。
def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_dir).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"official VQA Visual Robustness repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import generator as official_generator

    severity = int(args.severity)
    if not 1 <= severity <= 5:
        raise ValueError("severity must be in [1, 5]")
    seed = int(args.seed)
    np.random.seed(seed)

    official_key, params = _official_key(args.corruption_type, severity)
    dataset = _SingleImageDataset(Path(args.input_image).expanduser().resolve())
    gen = official_generator.Generator(dataset, _Logger(Path(args.output_image).expanduser().resolve().parent))
    if official_key not in gen.validTransformations:
        raise ValueError(f"unsupported official corruption {official_key!r}; available={sorted(gen.validTransformations)[:80]}")

    method_spec = gen.validTransformations[official_key]
    if isinstance(method_spec, tuple):
        method, official_severity = method_spec
        adv = method(0, official_severity)
        params["official_method_severity"] = official_severity
    else:
        adv = method_spec(0)

    output_image = Path(args.output_image).expanduser().resolve()
    _save_image(output_image, adv)
    trace = {
        "repo_dir": str(repo),
        "source": "https://github.com/ishmamt/VQA-Visual-Robustness-Benchmark",
        "official_module": "generator.Generator",
        "requested_corruption_type": args.corruption_type,
        "official_transformation": official_key,
        "severity": severity,
        "seed": seed,
        "actual_params": params,
        "output_image": str(output_image),
    }
    if args.trace_json:
        trace_path = Path(args.trace_json).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return trace


# 作为 `vqa_visual_corruption_one.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    parser = argparse.ArgumentParser(description="Run one official VQA Visual Robustness Benchmark corruption.")
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--output_image", required=True)
    parser.add_argument("--corruption_type", default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace_json", default="")
    args = parser.parse_args()
    trace = run(args)
    print(json.dumps(trace, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
