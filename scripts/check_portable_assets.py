# 文件说明：该文件属于运维与实验脚本，集中实现 check portable assets 相关逻辑。
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_LOCAL_DIRS


REQUIRED_HF_MODELS: tuple[str, ...] = (
    "clip",
    "blip_itm",
    "vilt_itm",
    "bert_mlm",
)

REQUIRED_LOCAL_VLM_MODELS: tuple[str, ...] = LOCAL_OPENAI_COMPAT_LOCAL_DIRS

WEIGHT_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "pytorch_model*.bin",
    "model*.bin",
)


# 中文注释：封装 _weight_files 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _weight_files(root: Path) -> list[str]:
    found: list[str] = []
    for pattern in WEIGHT_PATTERNS:
        for item in root.rglob(pattern):
            if item.is_file():
                found.append(str(item.relative_to(root)).replace("\\", "/"))
    return sorted(set(found))


# 中文注释：封装 _model_entry 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _model_entry(model_root: Path) -> dict[str, object]:
    config_exists = (model_root / "config.json").is_file()
    weights = _weight_files(model_root)
    return {
        "path": str(model_root),
        "exists": model_root.is_dir(),
        "config_exists": config_exists,
        "weight_files": weights,
        "ready": bool(model_root.is_dir() and config_exists and weights),
    }


# 中文注释：封装 _group_summary 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _group_summary(root: Path, names: Iterable[str]) -> dict[str, dict[str, object]]:
    return {str(name): _model_entry(root / str(name)) for name in names}


# 中文注释：封装 _missing_models 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _missing_models(summary: dict[str, dict[str, object]]) -> list[str]:
    return [name for name, row in summary.items() if not bool(row.get("ready", False))]


# 中文注释：实现 build_summary 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def build_summary(artifacts_root: Path) -> dict[str, object]:
    hf_root = artifacts_root / "hf_models"
    local_vlm_root = artifacts_root / "local_vlm"
    hf_summary = _group_summary(hf_root, REQUIRED_HF_MODELS)
    local_vlm_summary = _group_summary(local_vlm_root, REQUIRED_LOCAL_VLM_MODELS)
    return {
        "artifacts_root": str(artifacts_root),
        "hf_root": str(hf_root),
        "local_vlm_root": str(local_vlm_root),
        "required_hf_models": list(REQUIRED_HF_MODELS),
        "required_local_vlm_models": list(REQUIRED_LOCAL_VLM_MODELS),
        "hf_models": hf_summary,
        "local_vlm_models": local_vlm_summary,
        "missing_hf_models": _missing_models(hf_summary),
        "missing_local_vlm_models": _missing_models(local_vlm_summary),
    }


# 中文注释：串联 main 的主流程，集中处理运维与实验脚本的初始化、执行和退出条件。
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that the portable ATT-project image can be built fully offline."
    )
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    summary = build_summary(Path(args.artifacts_root).resolve())
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    missing = list(summary["missing_hf_models"]) + list(summary["missing_local_vlm_models"])
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
