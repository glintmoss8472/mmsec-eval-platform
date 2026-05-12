# 文件说明：该文件属于项目工程，集中实现 runtime 相关逻辑。
from __future__ import annotations

import logging
import os
import platform
from typing import Any

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.exceptions import ConfigError
from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_MODEL_SPECS

LOG = logging.getLogger(__name__)


# 判断 `真值` 输入是否表示真值，兼容字符串、数字和布尔类型。
def _truthy(value: str) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes", "on"}


# 执行 `detect PyTorch` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def _detect_torch() -> tuple[Any | None, str]:
    try:
        import torch  # type: ignore

        return torch, ""
    except (ImportError, OSError) as e:  # pragma: no cover
        return None, str(e)


# 执行 `PyTorch 是否包含 CUDA` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def _torch_has_cuda(torch_mod: Any) -> bool:
    try:
        if getattr(torch_mod, "version", None) is None:
            return False
        if getattr(torch_mod.version, "cuda", None) is None:
            # CPU-only builds report cuda=None.
            return False
        return bool(torch_mod.cuda.is_available())
    except (AttributeError, RuntimeError):  # pragma: no cover
        return False


# 执行 `PyTorch install command` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def torch_install_command() -> str:
    if platform.system().lower().startswith("win"):
        return "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_torch_cuda.ps1"
    return "bash scripts/install_torch_cuda.sh"


# 执行 `PyTorch install hint` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def torch_install_hint() -> str:
    return f"Run `{torch_install_command()}` and re-run."


# 解析 `runtime device` 的真实位置或配置值，减少调用方重复分支。
def resolve_runtime_device(cfg: AppConfig) -> str:
    """Resolve the actual runtime device for the current run.

    This project runs in GPU-only mode. The resolved device is always "cuda" or "cuda:N".
    """
    requested = str(cfg.runtime.device or "").strip()
    if not requested:
        raise ConfigError("runtime.device must be set to 'cuda' or 'cuda:0' (GPU-only).")
    req = requested.lower()
    if req in {"auto", "cpu"} or (not req.startswith("cuda")):
        raise ConfigError("runtime.device must be 'cuda' or 'cuda:0' (GPU-only).")

    torch_mod, err = _detect_torch()
    if torch_mod is None:
        raise ConfigError(f"CUDA is required but torch is not installed/usable: {err}")

    if _torch_has_cuda(torch_mod):
        # Preserve explicit device index when provided (e.g. cuda:0).
        return requested

    # Provide an actionable error message: most common is CPU-only torch wheel.
    torch_ver = str(getattr(torch_mod, "__version__", "unknown"))
    cuda_ver = str(getattr(getattr(torch_mod, "version", None), "cuda", None))
    raise ConfigError(
        "CUDA is required but torch CUDA is unavailable. "
        f"torch={torch_ver} torch.version.cuda={cuda_ver}. "
        f"Fix: {torch_install_hint()}"
    )


# 应用 `runtime 环境` 规则，把兼容字段写回报告或风险载荷。
def apply_runtime_env(cfg: AppConfig) -> str:
    """Apply runtime-related environment variables for downstream plugins.

    Returns the resolved device string.
    """
    device = resolve_runtime_device(cfg)
    os.environ["MMSEC_RUNTIME_DEVICE"] = str(device)
    os.environ["MMSEC_RUNTIME_AMP"] = "1" if bool(cfg.runtime.amp) else "0"
    os.environ["MMSEC_RUNTIME_DETERMINISTIC"] = "1" if bool(cfg.runtime.deterministic) else "0"
    os.environ["MMSEC_DEVICE_PREFERENCE"] = str(cfg.device_preference or "cuda")
    os.environ["MMSEC_MODEL_ENABLE_GRADIENTS"] = "1" if bool(cfg.model.enable_gradients) else "0"
    # Strict mode is always enabled (no fallbacks).
    os.environ["MMSEC_STRICT_REAL"] = "1"
    # Prefer offline-stable model loads by default; callers can override to "0".
    os.environ.setdefault("MMSEC_HF_LOCAL_ONLY", "1")
    # Disable background safetensors auto-conversion threads that may trigger flaky HF network calls.
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    return device


# 应用 `本地 视觉语言模型 环境 defaults` 规则，把兼容字段写回报告或风险载荷。
def apply_local_vlm_env_defaults(*, include_api_key_env: bool = False, include_timeout: bool = False) -> None:
    """Apply default environment variables for self-hosted OpenAI-compatible VLMs."""
    for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS:
        os.environ.setdefault(spec.model_name_env, spec.model_name_default)
        os.environ.setdefault(spec.endpoint_env, spec.endpoint_default)
        if include_api_key_env:
            os.environ.setdefault(spec.api_key_env, spec.api_key_env_default)
        if include_timeout:
            os.environ.setdefault(spec.timeout_env, spec.timeout_default)


# 应用 `配置 环境` 规则，把兼容字段写回报告或风险载荷。
def apply_config_env(cfg: AppConfig) -> str:
    """Apply runtime, model, judge, and adapter environment variables from config.

    This is the single entry point used by CLI and API job execution so both
    paths expose the same model names, endpoints, judge settings, and adapter
    timeouts to downstream plugins.
    """
    device = apply_runtime_env(cfg)

    if cfg.judge.llm_enabled:
        os.environ["MMSEC_LLM_JUDGE_ENABLED"] = "1"
        os.environ["MMSEC_LLM_PROVIDER"] = cfg.judge.llm_provider
        os.environ["MMSEC_LLM_ENDPOINT"] = cfg.judge.llm_endpoint
    else:
        os.environ["MMSEC_LLM_JUDGE_ENABLED"] = "0"

    os.environ["MMSEC_CLIP_MODEL_NAME"] = cfg.model.clip_model_name
    os.environ["MMSEC_BLIP_ITM_MODEL_NAME"] = cfg.model.blip_itm_model_name
    os.environ["MMSEC_VILT_ITM_MODEL_NAME"] = cfg.model.vilt_itm_model_name
    os.environ["MMSEC_OPENAI_COMPAT_MODEL_NAME"] = cfg.model.openai_model_name
    os.environ["MMSEC_OPENAI_COMPAT_BASE_URL"] = cfg.model.openai_base_url
    os.environ["MMSEC_OPENAI_COMPAT_API_KEY_ENV"] = cfg.model.openai_api_key_env
    os.environ["MMSEC_OPENAI_COMPAT_TIMEOUT"] = str(cfg.model.openai_timeout)
    apply_local_vlm_env_defaults()

    os.environ["MMSEC_GEMINI_MODEL_NAME"] = cfg.model.gemini_model_name
    os.environ["MMSEC_GEMINI_BASE_URL"] = cfg.model.gemini_base_url
    os.environ["MMSEC_GEMINI_API_KEY_ENV"] = cfg.model.gemini_api_key_env
    os.environ["MMSEC_GEMINI_TIMEOUT"] = str(cfg.model.gemini_timeout)
    if cfg.model.http_endpoint:
        os.environ["MMSEC_HTTP_ADAPTER_ENDPOINT"] = cfg.model.http_endpoint
    os.environ["MMSEC_HTTP_ADAPTER_RETRIES"] = str(cfg.model.http_retries)
    os.environ["MMSEC_HTTP_ADAPTER_TIMEOUT"] = str(cfg.model.http_timeout)
    return device


# 整理 `环境 runtime device`，描述当前服务器运行环境、模型入口或部署状态。
def env_runtime_device(default: str = "cuda") -> str:
    """Read the resolved runtime device from env (set by apply_runtime_env)."""
    v = str(os.getenv("MMSEC_RUNTIME_DEVICE", "")).strip()
    return v if v else default


# 执行 `环境 strict real` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def env_strict_real(default: bool = False) -> bool:
    v = os.getenv("MMSEC_STRICT_REAL", "")
    if v == "":
        return bool(default)
    return _truthy(v)


# 推断 `环境 模型 enable gradients`，从样本、配置或运行记录中提取统一名称。
def env_model_enable_gradients(default: bool = False) -> bool:
    v = os.getenv("MMSEC_MODEL_ENABLE_GRADIENTS", "")
    if v == "":
        return bool(default)
    return _truthy(v)
