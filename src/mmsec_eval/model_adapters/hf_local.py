# 文件说明：该文件属于模型适配层，集中实现 hf local 相关逻辑。
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mmsec_eval.runtime import env_runtime_device, torch_install_hint


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# 定位 `looks like Hugging Face 模型 目录`，把配置值或请求上下文转换成实际文件系统路径。
def _looks_like_hf_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (path / "pytorch_model.bin").exists() or any(path.glob("*.safetensors"))
    return bool(has_config and has_weights)


# 整理 `candidate Hugging Face 模型 目录` 路径信息，把本地文件或产物引用转换成统一表示。
def _candidate_hf_model_dirs(local_dir_name: str) -> list[Path]:
    candidates: list[Path] = []
    artifacts_dir = str(os.getenv("MMSEC_ARTIFACTS_DIR", "artifacts")).strip() or "artifacts"
    candidates.append(Path(artifacts_dir) / "hf_models" / str(local_dir_name))

    bundle_root = str(os.getenv("MMSEC_BUNDLE_ROOT", "") or "").strip()
    if bundle_root:
        candidates.append(Path(bundle_root) / "artifacts" / "hf_models" / str(local_dir_name))

    project_fallback = str(os.getenv("MMSEC_PROJECT_ARTIFACTS_FALLBACK", "1")).strip().lower()
    if project_fallback not in {"0", "false", "no", "off"}:
        candidates.append(PROJECT_ROOT / "artifacts" / "hf_models" / str(local_dir_name))

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


# 解析 `Hugging Face 模型 source` 的真实位置或配置值，减少调用方重复分支。
def resolve_hf_model_source(model_name: str, *, local_only: bool, local_dir_name: str) -> str:
    """
    Resolve model source for strict local-only mode.

    If local_only is true and artifacts/hf_models/<local_dir_name> exists with model files,
    return that local directory path so from_pretrained can load fully offline.
    """
    name = str(model_name or "").strip()
    if not local_only:
        return name

    for local_dir in _candidate_hf_model_dirs(local_dir_name):
        if _looks_like_hf_model_dir(local_dir):
            return str(local_dir)
    return name


# 定位 `Hugging Face 模型 本地 目录`，把配置值或请求上下文转换成实际文件系统路径。
def hf_model_local_dir(local_dir_name: str) -> Path:
    artifacts_dir = str(os.getenv("MMSEC_ARTIFACTS_DIR", "artifacts")).strip() or "artifacts"
    return Path(artifacts_dir) / "hf_models" / str(local_dir_name)


# 判断或归一 `Hugging Face 模型 ready` 状态，让调用方可以稳定渲染能力和可用性。
def hf_model_ready(model_name: str, *, local_only: bool, local_dir_name: str) -> tuple[str, bool]:
    source = resolve_hf_model_source(model_name, local_only=local_only, local_dir_name=local_dir_name)
    if not local_only:
        return source, True
    return source, bool(source != str(model_name or "").strip() and _looks_like_hf_model_dir(Path(source)))


# 执行 `Hugging Face load failure message` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def hf_load_failure_message(
    *,
    adapter_label: str,
    model_name: str,
    source: str,
    device: str,
    local_only: bool,
    cause: Exception,
) -> str:
    hint = (
        "Model not found in local cache. Prefetch first with scripts/reproduce_vlr_strong.ps1 "
        "or set MMSEC_HF_LOCAL_ONLY=0 to allow online download."
        if local_only
        else "Check model id/network/HF auth, or prefetch locally then run with MMSEC_HF_LOCAL_ONLY=1."
    )
    return (
        f"Failed to load {adapter_label} model '{model_name}' "
        f"(source={source}) on {device}. {hint} Root cause: {cause}"
    )


# 执行 `require CUDA device` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def require_cuda_device(adapter_label: str, torch_module: Any) -> str:
    requested = str(env_runtime_device(default="cuda")).strip() or "cuda"
    if not requested.lower().startswith("cuda"):
        raise RuntimeError(f"{adapter_label} requires CUDA runtime device, got: {requested}")
    if getattr(getattr(torch_module, "version", None), "cuda", None) is None or not torch_module.cuda.is_available():
        raise RuntimeError(
            f"{adapter_label} requires CUDA-enabled torch and an available GPU. "
            f"Fix: {torch_install_hint()}"
        )
    return requested
