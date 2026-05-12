# 文件说明：该文件属于自动化测试，集中实现 api test utils 相关逻辑。
from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


# 构建 `client` 数据，集中整理自动化测试需要的输出结构。
def make_client(tmp_path: Path, monkeypatch, *, skip_model_preflight: bool = False) -> TestClient:
    art = tmp_path / "artifacts"
    monkeypatch.setenv("MMSEC_ARTIFACTS_DIR", str(art))
    monkeypatch.setenv("MMSEC_APP_DB", str(art / "app.db"))
    monkeypatch.setenv("MMSEC_BOOTSTRAP_ENABLED", "0")
    if skip_model_preflight:
        monkeypatch.setenv("MMSEC_SKIP_MODEL_PREFLIGHT", "1")

    import mmsec_api.main as api_main

    importlib.reload(api_main)
    return TestClient(api_main.app)


# 执行 `wait 任务 done` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def wait_job_done(client: TestClient, job_id: str, timeout_s: float = 240.0) -> str:
    deadline = time.time() + float(timeout_s)
    status = "queued"
    while time.time() < deadline:
        cur = client.get(f"/api/v1/jobs/{job_id}")
        assert cur.status_code == 200
        status = str(cur.json()["status"])
        if status in {"success", "failed", "cancelled"}:
            return status
        time.sleep(0.2)
    return status


# 写出 `toy 评测 配置`，保证后续报告、页面或复现实验能读取。
def write_toy_eval_cfg(
    path: Path,
    *,
    attack: str = "advedm",
    num_samples: int = 2,
    image_size: int = 64,
    attack_config: dict[str, Any] | None = None,
    runner: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
) -> None:
    import yaml

    cfg: dict[str, Any] = {
        "seed": 1,
        "artifacts_dir": str(path.parent / "artifacts"),
        "plugins": {
            "model_adapter": "clip_hf",
            "attack": attack,
            "metric": "basic",
            "judge": "rule",
        },
        "dataset": {
            "kind": "toy_shapes",
            "num_samples": int(num_samples),
            "image_size": int(image_size),
        },
        "attack": dict(attack_config or {"steps": 1, "patch_size": 8}),
        "runner": dict(runner or {"max_samples": int(num_samples), "continue_on_error": False}),
    }
    if task:
        cfg["task"] = dict(task)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
