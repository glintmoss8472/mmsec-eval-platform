from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from mmsec_eval.model_adapters.local_vlm_catalog import (
    LOCAL_OPENAI_COMPAT_ADAPTERS,
    LOCAL_OPENAI_COMPAT_MODEL_SPECS,
    LocalVLMModelSpec,
    local_vlm_spec_by_adapter,
)

LOG_FN = Callable[[str], None]
_LOCAL_ADAPTERS = set(LOCAL_OPENAI_COMPAT_ADAPTERS)


def project_root_default() -> Path:
    return Path(__file__).resolve().parents[3]


def local_vlm_adapters(adapters: Iterable[str] | None) -> list[str]:
    seen: dict[str, None] = {}
    for item in adapters or []:
        adapter = str(item or "").strip()
        if adapter and adapter in _LOCAL_ADAPTERS:
            seen.setdefault(adapter, None)
    return list(seen.keys())


def has_local_vlm_adapter(adapters: Iterable[str] | None) -> bool:
    return bool(local_vlm_adapters(adapters))


def _health_url(endpoint: str) -> str:
    parts = urlsplit(str(endpoint or ""))
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    path = f"{path}/health" if path else "/health"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _is_ready(spec: LocalVLMModelSpec, *, timeout: float = 1.5) -> bool:
    url = _health_url(spec.endpoint_default)
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except (OSError, URLError, TimeoutError, ValueError):
        return False


def _scan_processes() -> list[tuple[int, str]]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    out: list[tuple[int, str]] = []
    for line in str(proc.stdout or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        pid_text, _, cmd = raw.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        out.append((pid, cmd.strip()))
    return out


def _pid_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def _target_specs(adapters: Iterable[str] | None) -> list[LocalVLMModelSpec]:
    names = local_vlm_adapters(adapters) if adapters is not None else list(LOCAL_OPENAI_COMPAT_ADAPTERS)
    return [local_vlm_spec_by_adapter(name) for name in names]


def stop_local_vlm_servers(
    *,
    adapters: Iterable[str] | None = None,
    log: LOG_FN | None = None,
    grace_seconds: float = 10.0,
) -> dict[str, Any]:
    """Stop project-managed local VLM HTTP servers to release GPU memory.

    If adapters is None, all known local VLM ports are targeted. This is used
    before attack generation because a leftover Qwen/InternVL/Gemma server can
    consume most of the single 4090 even when it is not part of the new run.
    """
    specs = _target_specs(adapters)
    ports = {int(spec.endpoint_port) for spec in specs}
    script_names = {Path(spec.launch_script).name for spec in specs}
    own_pid = os.getpid()
    matched: list[int] = []

    for pid, cmd in _scan_processes():
        if pid == own_pid:
            continue
        server_match = "local_openai_mm_server.py" in cmd and any(f"--port {port}" in cmd for port in ports)
        launcher_match = any(name and name in cmd for name in script_names) and ("bash" in cmd or "/bin/sh" in cmd)
        if server_match or launcher_match:
            matched.append(pid)

    stopped: list[int] = []
    for pid in sorted(set(matched)):
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            if log:
                log(f"no permission to stop local VLM pid={pid}")

    deadline = time.time() + max(0.5, float(grace_seconds))
    while time.time() < deadline and any(_pid_alive(pid) for pid in stopped):
        time.sleep(0.2)

    killed: list[int] = []
    for pid in stopped:
        if not _pid_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            continue

    if log and stopped:
        log(f"stopped local VLM servers pids={stopped} ports={sorted(ports)}")
    return {"ports": sorted(ports), "stopped_pids": stopped, "killed_pids": killed}


def empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except (AttributeError, RuntimeError):
        return


def _timeout_seconds(default: int = 1200) -> int:
    raw = str(os.getenv("MMSEC_MODEL_PREFLIGHT_TIMEOUT_SECONDS", str(default)) or "").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return int(default)


def _launch_local_vlm(spec: LocalVLMModelSpec, *, project_root: Path, log: LOG_FN | None = None) -> int:
    script_path = project_root / spec.launch_script
    if not script_path.exists():
        raise RuntimeError(f"本地 VLM 启动脚本不存在：{script_path}")
    env = os.environ.copy()
    env.setdefault("MMSEC_LOCAL_VLM_SINGLE_TENANT", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = str(proc.stderr or proc.stdout or "").strip() or "model launch failed"
        raise RuntimeError(f"本地 VLM 启动失败：{spec.display_name}；{detail}")
    pid = 0
    for line in str(proc.stdout or "").splitlines()[::-1]:
        try:
            pid = int(line.strip())
            break
        except ValueError:
            continue
    if log:
        log(f"launched local VLM adapter={spec.adapter} pid={pid} script={script_path}")
    return pid


def ensure_local_vlm_adapters_ready(
    adapters: Iterable[str] | None,
    *,
    project_root: Path | None = None,
    timeout_seconds: int | None = None,
    log: LOG_FN | None = None,
) -> dict[str, Any]:
    names = local_vlm_adapters(adapters)
    if not names:
        return {"local_vlm_adapters": [], "ready": True}
    if len(names) > 1:
        raise RuntimeError(
            "当前 staged 攻击/评测调度只支持一次运行中启动一个本地 VLM。"
            f"请在单卡 4090 上选择单个受测模型；当前选择：{', '.join(names)}"
        )

    project_root = project_root or project_root_default()
    timeout = int(timeout_seconds if timeout_seconds is not None else _timeout_seconds())
    started: list[dict[str, Any]] = []
    for adapter in names:
        spec = local_vlm_spec_by_adapter(adapter)
        if not _is_ready(spec):
            pid = _launch_local_vlm(spec, project_root=project_root, log=log)
            started.append({"adapter": adapter, "pid": pid})
        deadline = time.time() + max(10, timeout)
        while time.time() < deadline:
            if _is_ready(spec):
                break
            if started and int(started[-1].get("pid") or 0) and not _pid_alive(int(started[-1]["pid"])):
                raise RuntimeError(f"本地 VLM 进程已退出且未就绪：{spec.display_name}")
            time.sleep(3)
        else:
            raise RuntimeError(f"本地 VLM 启动超时：{spec.display_name}")
    return {"local_vlm_adapters": names, "ready": True, "started": started}
