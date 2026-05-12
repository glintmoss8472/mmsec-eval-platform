# 文件说明：该文件属于评测运行器，集中实现 artifacts 相关逻辑。
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from mmsec_eval.io.jsonl_io import write_jsonl


# 中文注释：实现 new_run_id 的核心流程，支撑评测运行器中的业务语义和异常边界。
def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# 中文注释：实现 make_run_dir 的核心流程，支撑评测运行器中的业务语义和异常边界。
def make_run_dir(base_dir: str, run_id: str) -> str:
    run_dir = Path(base_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


# 中文注释：实现 write_summary 的核心流程，支撑评测运行器中的业务语义和异常边界。
def write_summary(run_dir: str, summary: dict[str, Any]) -> str:
    p = Path(run_dir) / "summary.json"
    p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


# 中文注释：实现 write_results 的核心流程，支撑评测运行器中的业务语义和异常边界。
def write_results(run_dir: str, rows: list[dict[str, Any]]) -> str:
    p = Path(run_dir) / "results.jsonl"
    write_jsonl(str(p), rows)
    return str(p)


# 中文注释：实现 write_json_snapshot 的核心流程，支撑评测运行器中的业务语义和异常边界。
def write_json_snapshot(run_dir: str, name: str, data: dict[str, Any]) -> str:
    p = Path(run_dir) / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


# 中文注释：实现 write_env_snapshot 的核心流程，支撑评测运行器中的业务语义和异常边界。
def write_env_snapshot(run_dir: str) -> str:
    keys = [
        "MMSEC_RUNTIME_DEVICE",
        "MMSEC_STRICT_REAL",
        "MMSEC_HF_LOCAL_ONLY",
        "DISABLE_SAFETENSORS_CONVERSION",
        "MMSEC_CLIP_MODEL_NAME",
        "MMSEC_BLIP_ITM_MODEL_NAME",
        "MMSEC_VILT_ITM_MODEL_NAME",
        "MMSEC_HTTP_ADAPTER_ENDPOINT",
        "MMSEC_HTTP_ADAPTER_RETRIES",
        "MMSEC_HTTP_ADAPTER_TIMEOUT",
        "MMSEC_LLM_JUDGE_ENABLED",
        "MMSEC_LLM_PROVIDER",
        "MMSEC_LLM_ENDPOINT",
    ]
    data = {k: os.getenv(k, "") for k in keys}
    return write_json_snapshot(run_dir, "env_snapshot.json", data)
