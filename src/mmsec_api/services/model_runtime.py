# 文件说明：该文件属于后端业务服务，集中实现 model runtime 相关逻辑。
from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from mmsec_api.utils import utc_now_iso
from mmsec_eval.model_adapters.hf_local import hf_model_ready
from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_ADAPTERS, LOCAL_OPENAI_COMPAT_MODEL_SPECS, LocalVLMModelSpec


# 中文注释：定义 ModelRuntimeSpec 的结构化职责，作为后端业务服务中状态、配置或行为的边界。
@dataclass(frozen=True)
class ModelRuntimeSpec:
    adapter: str
    display_name: str
    family: str
    launch_mode: str
    role: str
    model_name_env: str
    model_name_default: str
    endpoint_env: str | None = None
    endpoint_default: str | None = None
    hf_local_dir_name: str | None = None
    launch_script: str | None = None
    launch_log: str | None = None


# 中文注释：封装 _local_vlm_runtime_spec 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _local_vlm_runtime_spec(spec: LocalVLMModelSpec) -> ModelRuntimeSpec:
    return ModelRuntimeSpec(
        adapter=spec.adapter,
        display_name=spec.display_name,
        family="本地视觉语言模型",
        launch_mode="本地自托管服务",
        role="victim/api_or_self_hosted",
        model_name_env=spec.model_name_env,
        model_name_default=spec.model_name_default,
        endpoint_env=spec.endpoint_env,
        endpoint_default=spec.endpoint_default,
        launch_script=spec.launch_script,
        launch_log=spec.launch_log,
    )


# 中文注释：封装 _local_vlm_runtime_specs 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _local_vlm_runtime_specs() -> tuple[ModelRuntimeSpec, ...]:
    return tuple(_local_vlm_runtime_spec(spec) for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS)


MAIN_MODEL_SPECS: tuple[ModelRuntimeSpec, ...] = (
    ModelRuntimeSpec(
        adapter="clip_hf",
        display_name="对比语言图像预训练模型（CLIP）",
        family="经典检索模型",
        launch_mode="本地离线加载",
        role="surrogate/local",
        model_name_env="MMSEC_CLIP_MODEL_NAME",
        model_name_default="openai/clip-vit-base-patch32",
        hf_local_dir_name="clip",
    ),
    ModelRuntimeSpec(
        adapter="blip_itm",
        display_name="图文匹配模型（BLIP）",
        family="经典检索模型",
        launch_mode="本地离线加载",
        role="victim/local",
        model_name_env="MMSEC_BLIP_ITM_MODEL_NAME",
        model_name_default="Salesforce/blip-itm-base-coco",
        hf_local_dir_name="blip_itm",
    ),
    ModelRuntimeSpec(
        adapter="vilt_itm",
        display_name="视觉语言 Transformer 匹配模型（ViLT）",
        family="经典检索模型",
        launch_mode="本地离线加载",
        role="victim/local",
        model_name_env="MMSEC_VILT_ITM_MODEL_NAME",
        model_name_default="dandelin/vilt-b32-finetuned-coco",
        hf_local_dir_name="vilt_itm",
    ),
    *_local_vlm_runtime_specs(),
)


FORMAL_EXCLUDED_ADAPTERS = {"fixture_vlm"}
RETRIEVAL_MODEL_ADAPTERS = {
    "clip_hf",
    "blip_itm",
    "vilt_itm",
    "openai_compat",
    "openai_gpt4o",
    "gemini_vision",
    "http",
    *LOCAL_OPENAI_COMPAT_ADAPTERS,
}
GENERATION_MODEL_ADAPTERS = {
    "openai_compat",
    "openai_gpt4o",
    "gemini_vision",
    "http",
    *LOCAL_OPENAI_COMPAT_ADAPTERS,
}


# 中文注释：实现 task_capabilities_for_adapter 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def task_capabilities_for_adapter(adapter: str, *, formal: bool = True) -> list[str]:
    adapter_id = str(adapter or "").strip()
    if not adapter_id:
        return []
    if formal and adapter_id in FORMAL_EXCLUDED_ADAPTERS:
        return []
    capabilities: list[str] = []
    if adapter_id in RETRIEVAL_MODEL_ADAPTERS:
        capabilities.append("vlr")
    if adapter_id in GENERATION_MODEL_ADAPTERS:
        capabilities.extend(["vqa", "caption"])
    return capabilities


# 中文注释：实现 model_supports_task 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def model_supports_task(adapter: str, task_kind: str, *, formal: bool = True) -> bool:
    return str(task_kind or "").strip() in task_capabilities_for_adapter(adapter, formal=formal)


# 中文注释：实现 task_capability_note 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def task_capability_note(adapter: str) -> str:
    if str(adapter or "").strip() == "fixture_vlm":
        return "内置演示模型只用于开发 smoke，不参与真实测评任务选择。"
    return ""


# 中文注释：封装 _loopback_base_url 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _loopback_base_url(url: str) -> bool:
    host = str(urlparse(str(url or "")).hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost"}


# 中文注释：封装 _model_name 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _model_name(spec: ModelRuntimeSpec) -> str:
    return str(os.getenv(spec.model_name_env, spec.model_name_default)).strip() or spec.model_name_default


# 中文注释：封装 _endpoint 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _endpoint(spec: ModelRuntimeSpec) -> str:
    if not spec.endpoint_env:
        return ""
    return str(os.getenv(spec.endpoint_env, spec.endpoint_default or "")).strip()


# 中文注释：封装 _probe_openai_service 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _probe_openai_service(base_url: str) -> tuple[str, str]:
    root = str(base_url or "").rstrip("/")
    if not root:
        return "unavailable", ""
    session = requests.Session()
    if _loopback_base_url(root):
        session.trust_env = False
    last_error = ""
    try:
        resp = session.get(f"{root}/models", timeout=1.5)
        if resp.ok:
            return "ready", f"{root}/models"
    except requests.RequestException as exc:
        last_error = f"models probe failed: {type(exc).__name__}"
    if root.endswith("/v1"):
        health_url = f"{root[:-3]}/health"
    else:
        health_url = f"{root}/health"
    try:
        resp = session.get(health_url, timeout=1.5)
        if resp.ok:
            return "ready", health_url
    except requests.RequestException as exc:
        last_error = f"health probe failed: {type(exc).__name__}" or last_error
    return "unavailable", last_error or root


# 中文注释：封装 _local_only 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _local_only() -> bool:
    return str(os.getenv("MMSEC_HF_LOCAL_ONLY", "1")).strip().lower() not in {"0", "false", "no"}


# 中文注释：封装 _preflight_timeout_seconds 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _preflight_timeout_seconds(default: int = 1200) -> int:
    raw = str(os.getenv("MMSEC_MODEL_PREFLIGHT_TIMEOUT_SECONDS", str(default)) or "").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return int(default)


# 中文注释：封装 _launch_script_preflight 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _launch_script_preflight(spec: ModelRuntimeSpec, *, project_root: Path) -> tuple[str, str]:
    script_path = project_root / str(spec.launch_script or "")
    if not spec.launch_script or not script_path.exists():
        return "unavailable", f"missing model launch script: {script_path}"
    if platform.system().lower().startswith("win"):
        return "launch_blocked", "current host is not a Linux self-hosting runtime"
    try:
        proc = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(project_root),
            env={**os.environ, "MMSEC_MODEL_SERVER_PREFLIGHT": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "launch_blocked", f"launch preflight failed: {exc}"
    detail = str(proc.stderr or "").strip()
    if proc.returncode == 0:
        return "launchable", detail
    return "launch_blocked", detail or "launch preflight failed"


# 中文注释：封装 _launch_log_path 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _launch_log_path(spec: ModelRuntimeSpec, *, project_root: Path) -> Path | None:
    if not spec.launch_log:
        return None
    return project_root / "logs" / "model_servers" / spec.launch_log


# 中文注释：封装 _launch_log_tail 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _launch_log_tail(spec: ModelRuntimeSpec, *, project_root: Path, lines: int = 30) -> str:
    log_path = _launch_log_path(spec, project_root=project_root)
    if log_path is None or not log_path.exists():
        return ""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-max(1, int(lines)):]).strip()


# 中文注释：封装 _pid_alive 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    return Path(f"/proc/{pid}").exists()


# 中文注释：封装 _model_health 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
def _model_health(spec: ModelRuntimeSpec, *, project_root: Path | None = None) -> dict[str, Any]:
    last_checked_at = utc_now_iso()
    model_name = _model_name(spec)
    endpoint_or_source = ""
    health_status = "unavailable"
    health_detail = ""

    if spec.adapter == "fixture_vlm":
        endpoint_or_source = "builtin://fixture-vlm"
        health_status = "ready"
    elif spec.hf_local_dir_name:
        source, ready = hf_model_ready(model_name, local_only=_local_only(), local_dir_name=spec.hf_local_dir_name)
        endpoint_or_source = source
        health_status = "ready" if ready else "missing_assets"
    else:
        base_url = _endpoint(spec)
        status, detail = _probe_openai_service(base_url)
        endpoint_or_source = detail or base_url
        if status == "ready":
            health_status = "ready"
        elif project_root and spec.launch_script:
            health_status, health_detail = _launch_script_preflight(spec, project_root=project_root)
        else:
            script_path = Path(spec.launch_script) if spec.launch_script else None
            health_status = "launchable" if script_path and script_path.exists() else "unavailable"

    return {
        "adapter": spec.adapter,
        "display_name": spec.display_name,
        "family": spec.family,
        "launch_mode": spec.launch_mode,
        "health_status": health_status,
        "health_detail": health_detail,
        "last_checked_at": last_checked_at,
        "endpoint_or_source": endpoint_or_source,
        "model_name": model_name,
        "role": spec.role,
        "task_capabilities": task_capabilities_for_adapter(spec.adapter),
        "formal_eval": bool(task_capabilities_for_adapter(spec.adapter)),
        "capability_note": task_capability_note(spec.adapter),
    }


# 中文注释：实现 list_main_models 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def list_main_models(*, project_root: Path | None = None) -> list[dict[str, Any]]:
    return [_model_health(spec, project_root=project_root) for spec in MAIN_MODEL_SPECS]


# 中文注释：实现 build_adapter_env 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def build_adapter_env(models: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for item in models:
        out[str(item["adapter"])] = {
            "activation": item["health_status"],
            "model_name": item["model_name"],
            "role": item["role"],
            "endpoint": item["endpoint_or_source"],
        }
    return out


# 中文注释：实现 ensure_models_ready 的核心流程，支撑后端业务服务中的业务语义和异常边界。
def ensure_models_ready(
    adapters: list[str],
    *,
    project_root: Path,
    log: Callable[[str, str], None] | None = None,
    timeout_seconds: int | None = None,
) -> list[dict[str, Any]]:
    selected = {str(item) for item in adapters if str(item).strip()}
    if not selected:
        return list_main_models(project_root=project_root)

    current = {row["adapter"]: row for row in list_main_models(project_root=project_root)}
    for adapter in sorted(selected):
        row = current.get(adapter)
        if not row:
            raise RuntimeError(f"未找到受测模型适配器：{adapter}")
        if row["health_status"] == "ready":
            continue

        spec = next((item for item in MAIN_MODEL_SPECS if item.adapter == adapter), None)
        if spec is None:
            raise RuntimeError(f"未找到模型定义：{adapter}")
        if spec.hf_local_dir_name:
            raise RuntimeError(f"本地离线模型资产未就绪：{spec.display_name}")
        if not spec.launch_script:
            raise RuntimeError(f"模型没有可启动脚本：{spec.display_name}")
        if platform.system().lower().startswith("win"):
            raise RuntimeError(f"当前仅支持在 Linux 服务器上自动拉起本地视觉模型：{spec.display_name}")

        script_path = project_root / spec.launch_script
        if not script_path.exists():
            raise RuntimeError(f"模型启动脚本不存在：{script_path}")

        preflight_status, preflight_detail = _launch_script_preflight(spec, project_root=project_root)
        if preflight_status != "launchable":
            detail = preflight_detail or "model launch preflight failed"
            raise RuntimeError(f"模型当前不可自动拉起：{spec.display_name}；{detail}")

        if log:
            log("info", f"auto-start model server: adapter={adapter} script={script_path}")

        launch = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if launch.returncode != 0:
            detail = str(launch.stderr or launch.stdout or "").strip() or "model launch failed"
            raise RuntimeError(f"模型启动失败：{spec.display_name}；{detail}")

        launched_pid = 0
        pid_lines = [str(item).strip() for item in str(launch.stdout or "").splitlines() if str(item).strip()]
        if pid_lines:
            try:
                launched_pid = int(pid_lines[-1])
            except ValueError:
                launched_pid = 0

        wait_seconds = int(timeout_seconds if timeout_seconds is not None else _preflight_timeout_seconds())
        deadline = time.time() + max(10, wait_seconds)
        while time.time() < deadline:
            current = {item["adapter"]: item for item in list_main_models(project_root=project_root)}
            row = current.get(adapter)
            if row and row["health_status"] == "ready":
                break
            if launched_pid and not _pid_alive(launched_pid):
                log_tail = _launch_log_tail(spec, project_root=project_root)
                detail = f"；日志尾部：{log_tail}" if log_tail else ""
                raise RuntimeError(f"模型启动失败：{spec.display_name}{detail}")
            time.sleep(3)
        else:
            log_tail = _launch_log_tail(spec, project_root=project_root)
            detail = f"；日志尾部：{log_tail}" if log_tail else ""
            raise RuntimeError(f"模型启动后仍未就绪：{spec.display_name}{detail}")

    return list_main_models(project_root=project_root)
