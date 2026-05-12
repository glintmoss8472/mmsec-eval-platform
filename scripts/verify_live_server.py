# 文件说明：该文件属于运维与实验脚本，集中实现 verify live server 相关逻辑。
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_ADAPTERS


# 执行 `now tag` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


# 执行 `git value` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _git_value(repo: Path, args: list[str]) -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


# 执行 `本地 build fingerprint` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _local_build_fingerprint(deployment_root: Path) -> dict[str, Any]:
    version_path = deployment_root / "deployment_version.json"
    version_data = json.loads(version_path.read_text(encoding="utf-8")) if version_path.exists() else {}
    if not isinstance(version_data, dict):
        version_data = {}
    frontend_root = deployment_root / "frontend"
    src_dir = frontend_root / "src"
    index_path = frontend_root / "dist" / "index.html"
    if not index_path.exists():
        return {
            "deployment_target": str(version_data.get("deployment_target", "") or ""),
            "backend_version_stamp": str(version_data.get("version", "") or ""),
            "version_source": "deployment_version.json" if version_path.exists() else "",
            "backend_commit": _git_value(deployment_root, ["rev-parse", "--short", "HEAD"]),
            "frontend_index_exists": False,
            "frontend_dist_fresh": False,
            "frontend_asset_refs": [],
        }

    index_text = index_path.read_text(encoding="utf-8")
    asset_refs = sorted({match.group(1) for match in re.finditer(r'(?:src|href)="(/assets/[^"]+)"', index_text)})
    index_mtime = index_path.stat().st_mtime
    build_inputs = [
        frontend_root / "index.html",
        frontend_root / "vite.config.ts",
        frontend_root / "tailwind.config.cjs",
        frontend_root / "package.json",
        frontend_root / "pnpm-lock.yaml",
    ]
    input_times = [item.stat().st_mtime for item in src_dir.rglob("*") if item.is_file()]
    input_times.extend(path.stat().st_mtime for path in build_inputs if path.exists())
    src_latest_mtime = max(input_times, default=0.0)
    return {
        "deployment_target": str(version_data.get("deployment_target", "") or ""),
        "backend_version_stamp": str(version_data.get("version", "") or ""),
        "version_source": "deployment_version.json" if version_path.exists() else "",
        "backend_commit": _git_value(deployment_root, ["rev-parse", "--short", "HEAD"]),
        "frontend_index_exists": True,
        "frontend_dist_fresh": bool(index_mtime >= src_latest_mtime) if src_latest_mtime else True,
        "frontend_index_built_at": datetime.fromtimestamp(index_mtime, tz=UTC).isoformat(),
        "frontend_inputs_latest_at": datetime.fromtimestamp(src_latest_mtime, tz=UTC).isoformat() if src_latest_mtime else "",
        "frontend_asset_refs": asset_refs,
    }


# 推断 `数据集 override`，从样本、配置或运行记录中提取统一名称。
def _dataset_override(name: str) -> dict[str, Any]:
    if name == "mini_flickr":
        return {
            "kind": "mini_flickr",
            "root": "",
            "image_dir": "images",
            "captions_file": "captions_index.jsonl",
            "split": "test",
            "benchmark_tag": "mini_flickr",
        }
    if name == "flickr30k":
        return {
            "kind": "flickr30k",
            "root": "data/flickr30k",
            "image_dir": "images",
            "captions_file": "captions_index.jsonl",
            "split": "test",
            "benchmark_tag": "flickr30k",
        }
    if name == "coco_subset":
        return {
            "kind": "coco_subset",
            "root": "data/coco",
            "image_dir": "val2017",
            "captions_file": "annotations/captions_val2017_subset.json",
            "split": "val",
            "benchmark_tag": "coco_subset",
        }
    raise KeyError(f"unsupported dataset: {name}")


# 执行 `基础 override` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _base_override(*, attack: str, victims: list[str], dataset_name: str, experiment_id: str) -> dict[str, Any]:
    eval_scope = "joint" if attack in {"tmm", "advedm_plus"} else "image"
    return {
        "plugins": {
            "attack": attack,
            "model_adapter": "clip_hf",
        },
        "dataset": {
            **_dataset_override(dataset_name),
            "benchmark_tag": f"{experiment_id}_{dataset_name}",
            "max_items": 8 if dataset_name == "mini_flickr" else 16,
        },
        "task": {
            "kind": "vlr",
            "eval_scope": eval_scope,
        },
        "runner": {
            "surrogate_model_adapter": "clip_hf",
            "victim_model_adapters": victims,
            "max_pairs": 16,
            "experiment_id": experiment_id,
            "save_plots": False,
        },
        "report": {
            "save_heatmaps": True,
            "save_patch_preview": True,
            "top_k_cases": 6,
        },
        "sample_store": {
            "enabled": True,
            "save_images": True,
            "save_traces": True,
        },
        "defense": {
            "enabled": False,
            "apply_on_clean": False,
            "apply_on_attacked": False,
        },
        "attack": {
            "epsilon": 0.05,
            "step_size": 0.01,
            "steps": 8,
            "patch_size": 16,
            "patch_train_steps": 80,
            "eps_t": 1,
            "text_candidates_k": 8,
        },
    }


# 定义 `AttackSpec` 的状态和行为边界，供运维与实验脚本在固定职责内复用。
@dataclass(frozen=True)
class AttackSpec:
    attack: str
    config_path: str
    category: str


ATTACK_SPECS: tuple[AttackSpec, ...] = (
    AttackSpec("advclip", "configs/bench/bootstrap_full_vlr_cuda.yaml", "paper"),
    AttackSpec("tmm", "configs/bench/bootstrap_full_vlr_tmm_cuda.yaml", "paper"),
    AttackSpec("advedm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "paper"),
    AttackSpec("advedm_plus", "configs/bench/bootstrap_full_vlr_advedm_plus_cuda.yaml", "improved"),
    AttackSpec("fgsm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("bim", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("pgd", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("mifgsm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("nifgsm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("difgsm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("tifgsm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("dtmifgsm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("vmifgsm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("vnifgsm", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
    AttackSpec("cw", "configs/bench/bootstrap_full_vlr_cuda.yaml", "baseline"),
)


MAIN_MODELS: tuple[str, ...] = (
    "clip_hf",
    "blip_itm",
    "vilt_itm",
    *LOCAL_OPENAI_COMPAT_ADAPTERS,
)


# 实现 `ApiClient.__init__` 的对象行为，维护该类在运维与实验脚本中的调用契约。
class ApiClient:
    # 封装 ApiClient.__init__ 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
    def __init__(self, api_base: str, *, timeout: float = 60.0) -> None:
        self.api_base = str(api_base).rstrip("/")
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.session.trust_env = False

    # 实现 `ApiClient.get` 的对象行为，维护该类在运维与实验脚本中的调用契约。
    def get(self, path: str) -> Any:
        resp = self.session.get(f"{self.api_base}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # 实现 `ApiClient.post` 的对象行为，维护该类在运维与实验脚本中的调用契约。
    def post(self, path: str, payload: dict[str, Any]) -> Any:
        resp = self.session.post(f"{self.api_base}{path}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # 创建 `任务`，初始化后续流程所需的记录、对象或产物。
    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.post("/jobs", payload)

    # 获取 `任务`，封装存储查询或状态读取细节。
    def get_job(self, job_id: str) -> dict[str, Any]:
        return self.get(f"/jobs/{job_id}")

    # 获取 `任务 日志`，封装存储查询或状态读取细节。
    def get_job_logs(self, job_id: str, *, page_size: int = 200) -> dict[str, Any]:
        return self.get(f"/jobs/{job_id}/logs?page=1&page_size={page_size}")

    # 列出 `运行记录`，按调用方需要组织分页或过滤结果。
    def list_runs(self, *, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        return self.get(f"/runs?page={page}&page_size={page_size}")

    # 获取 `运行记录 摘要`，封装存储查询或状态读取细节。
    def get_run_summary(self, run_id: str) -> dict[str, Any]:
        return self.get(f"/runs/{run_id}/summary")

    # 获取 `运行记录 报告 数据`，封装存储查询或状态读取细节。
    def get_run_report_data(self, run_id: str) -> dict[str, Any]:
        return self.get(f"/runs/{run_id}/report-data")

    # 获取 `运行记录 案例`，封装存储查询或状态读取细节。
    def get_run_cases(self, run_id: str, *, page_size: int = 10) -> dict[str, Any]:
        return self.get(f"/runs/{run_id}/cases?page=1&page_size={page_size}")


# 执行 `wait 所属 任务` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _wait_for_job(
    api: ApiClient,
    job_id: str,
    *,
    timeout_seconds: int,
    poll_seconds: float = 3.0,
) -> dict[str, Any]:
    deadline = time.time() + max(30, int(timeout_seconds))
    while time.time() < deadline:
        job = api.get_job(job_id)
        status = str(job.get("status", ""))
        if status in {"success", "failed", "cancelled"}:
            return job
        time.sleep(poll_seconds)
    raise TimeoutError(f"job timed out: {job_id}")


# 执行 `verify 系统总览` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _verify_overview(api: ApiClient, *, deployment_root: Path, deployment_name: str) -> dict[str, Any]:
    overview = api.get("/system/overview")
    models = overview.get("models", [])
    datasets = overview.get("datasets", [])
    attacks = overview.get("attacks", [])
    states = {str(row.get("adapter", "")): str(row.get("health_status", "")) for row in models}
    server_build = dict(overview.get("build_identity", {}))
    local_build = _local_build_fingerprint(deployment_root)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "deployment_name": deployment_name,
        "deployment_root": str(deployment_root),
        "counts": {
            "attacks": len(attacks),
            "models": len(models),
            "datasets": len(datasets),
        },
        "states": states,
        "ready_models": sorted([name for name, state in states.items() if state == "ready"]),
        "launchable_models": sorted([name for name, state in states.items() if state == "launchable"]),
        "build_identity": server_build,
        "expected_build_identity": local_build,
        "raw": overview,
        "checks": {
            "attack_count_ok": len(attacks) >= 15,
            "model_count_ok": len(models) >= 10,
            "dataset_count_ok": len(datasets) >= 3,
            "main_models_present": all(name in states for name in MAIN_MODELS),
            "backend_version_matches": str(server_build.get("backend_version_stamp", "")) == str(local_build.get("backend_version_stamp", "")),
            "deployment_target_matches": str(server_build.get("deployment_target", "")) == str(local_build.get("deployment_target", "")),
            "backend_commit_matches": (
                not str(server_build.get("backend_commit", ""))
                or not str(local_build.get("backend_commit", ""))
                or str(server_build.get("backend_commit", "")) == str(local_build.get("backend_commit", ""))
            ),
            "frontend_assets_match": list(server_build.get("frontend_asset_refs", [])) == list(local_build.get("frontend_asset_refs", [])),
            "frontend_dist_fresh": bool(server_build.get("frontend_dist_fresh", False)),
        },
    }


# 构建 `运行记录 攻击 矩阵`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _run_attack_matrix(
    api: ApiClient,
    *,
    out_dir: Path,
    timeout_seconds: int,
    attack_specs: tuple[AttackSpec, ...],
) -> dict[str, Any]:
    timestamp = _now_tag()
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(attack_specs, start=1):
        result = _run_attack_matrix_spec(api, spec, index=index, timestamp=timestamp, timeout_seconds=timeout_seconds)
        rows.append(result)
        (out_dir / "attack_matrix.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    success_count = sum(1 for row in rows if row.get("job_status") == "success")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "count": len(rows),
        "success_count": success_count,
        "failed_count": len(rows) - success_count,
        "all_success": success_count == len(rows),
        "rows": rows,
    }


# 构建 `运行记录 攻击 矩阵 spec`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _run_attack_matrix_spec(
    api: ApiClient,
    spec: AttackSpec,
    *,
    index: int,
    timestamp: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = _attack_matrix_payload(spec, timestamp)
    job = api.create_job(payload)
    result = _attack_matrix_seed_result(spec, index, str(job.get("id", "")))
    try:
        final_job = _wait_for_job(api, result["job_id"], timeout_seconds=timeout_seconds)
        result["job_status"] = str(final_job.get("status", ""))
        result["finished_at"] = datetime.now(UTC).isoformat()
        if result["job_status"] == "success":
            result.update(_successful_attack_artifacts(api, str(final_job.get("run_id", "")).strip()))
        else:
            result["logs_tail"] = api.get_job_logs(result["job_id"], page_size=200).get("items", [])[-20:]
    except (KeyError, OSError, RuntimeError, TimeoutError, TypeError, ValueError, requests.RequestException) as exc:
        result["job_status"] = "failed"
        result["exception"] = str(exc)
        try:
            result["logs_tail"] = api.get_job_logs(result["job_id"], page_size=200).get("items", [])[-20:]
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, requests.RequestException) as log_exc:
            result["logs_error"] = str(log_exc)
    return result


# 构建 `攻击 矩阵 载荷`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _attack_matrix_payload(spec: AttackSpec, timestamp: str) -> dict[str, Any]:
    override = _base_override(
        attack=spec.attack,
        victims=["clip_hf"],
        dataset_name="mini_flickr",
        experiment_id=f"verify_attack_{spec.attack}_{timestamp}",
    )
    if spec.attack == "advedm":
        override["plugins"]["attack"] = "advedm"
    return {"job_type": "run_vlr", "config_path": spec.config_path, "override": override, "benchmark_mode": False}


# 构建 `攻击 矩阵 seed result`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _attack_matrix_seed_result(spec: AttackSpec, index: int, job_id: str) -> dict[str, Any]:
    return {
        "index": index,
        "attack": spec.attack,
        "category": spec.category,
        "job_id": job_id,
        "started_at": datetime.now(UTC).isoformat(),
    }


# 推断 `successful 攻击 产物`，从样本、配置或运行记录中提取统一名称。
def _successful_attack_artifacts(api: ApiClient, run_id: str) -> dict[str, Any]:
    summary = api.get_run_summary(run_id)
    report = api.get_run_report_data(run_id)
    cases = api.get_run_cases(run_id, page_size=3)
    return {
        "run_id": run_id,
        "summary": {
            "attack": summary.get("attack"),
            "dataset_name": summary.get("dataset_name"),
            "victim_model_adapters": summary.get("victim_model_adapters", []),
            "asr_attack": summary.get("asr_attack", summary.get("asr", 0.0)),
            "risk_score": summary.get("risk_score", 0.0),
            "num_victim_failures": summary.get("num_victim_failures", 0),
        },
        "artifacts": {
            "has_report_data": bool(report),
            "case_count": int(cases.get("total", 0) or 0),
            "top_case_ids": [str(item.get("sample_id", "")) for item in cases.get("items", [])[:3]],
        },
    }


# 构建 `运行记录 模型 矩阵`，把图像和文本两两配对后整理成指标计算所需的二维结果。
def _run_model_matrix(
    api: ApiClient,
    *,
    out_dir: Path,
    timeout_seconds: int,
    model_adapters: tuple[str, ...],
) -> dict[str, Any]:
    timestamp = _now_tag()
    rows: list[dict[str, Any]] = []
    for index, model_adapter in enumerate(model_adapters, start=1):
        experiment_id = f"verify_model_{model_adapter}_{timestamp}"
        override = _base_override(
            attack="fgsm",
            victims=[model_adapter],
            dataset_name="mini_flickr",
            experiment_id=experiment_id,
        )
        override["attack"]["steps"] = 1
        payload = {
            "job_type": "run_vlr",
            "config_path": "configs/bench/bootstrap_full_vlr_cuda.yaml",
            "override": override,
            "benchmark_mode": False,
        }
        started_at = datetime.now(UTC).isoformat()
        job = api.create_job(payload)
        result: dict[str, Any] = {
            "index": index,
            "model_adapter": model_adapter,
            "job_id": str(job.get("id", "")),
            "started_at": started_at,
        }
        try:
            final_job = _wait_for_job(api, result["job_id"], timeout_seconds=timeout_seconds, poll_seconds=4.0)
            result["job_status"] = str(final_job.get("status", ""))
            result["finished_at"] = datetime.now(UTC).isoformat()
            if result["job_status"] == "success":
                run_id = str(final_job.get("run_id", "")).strip()
                result["run_id"] = run_id
                summary = api.get_run_summary(run_id)
                result["summary"] = {
                    "attack": summary.get("attack"),
                    "dataset_name": summary.get("dataset_name"),
                    "victim_model_adapters": summary.get("victim_model_adapters", []),
                    "asr_attack": summary.get("asr_attack", summary.get("asr", 0.0)),
                    "risk_score": summary.get("risk_score", 0.0),
                    "num_victim_failures": summary.get("num_victim_failures", 0),
                }
            else:
                logs = api.get_job_logs(result["job_id"], page_size=200)
                result["logs_tail"] = logs.get("items", [])[-20:]
        except (KeyError, OSError, RuntimeError, TimeoutError, TypeError, ValueError, requests.RequestException) as exc:
            result["job_status"] = "failed"
            result["exception"] = str(exc)
            try:
                logs = api.get_job_logs(result["job_id"], page_size=200)
                result["logs_tail"] = logs.get("items", [])[-20:]
            except (KeyError, OSError, RuntimeError, TypeError, ValueError, requests.RequestException) as log_exc:
                result["logs_error"] = str(log_exc)
        rows.append(result)
        (out_dir / "model_matrix.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    success_count = sum(1 for row in rows if row.get("job_status") == "success")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "count": len(rows),
        "success_count": success_count,
        "failed_count": len(rows) - success_count,
        "all_success": success_count == len(rows),
        "rows": rows,
    }


# 作为 `verify_live_server.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:18081/api/v1")
    parser.add_argument("--out-dir", default="artifacts/verification/live_server")
    parser.add_argument("--skip-attacks", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--deployment", choices=("main", "shixian"), default="main")
    parser.add_argument("--attack-ids", default="")
    parser.add_argument("--model-adapters", default="")
    parser.add_argument("--attack-timeout", type=int, default=1800)
    parser.add_argument("--model-timeout", type=int, default=3600)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    api = ApiClient(args.api_base)
    deployment_root = REPO_ROOT if args.deployment == "main" else REPO_ROOT / "shixian"

    overview = _verify_overview(api, deployment_root=deployment_root, deployment_name=args.deployment)
    (out_dir / "system_overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "api_base": args.api_base,
        "deployment": args.deployment,
        "overview": overview,
    }

    selected_attack_ids = {item.strip() for item in str(args.attack_ids or "").split(",") if item.strip()}
    selected_model_adapters = {item.strip() for item in str(args.model_adapters or "").split(",") if item.strip()}
    attack_specs = tuple(item for item in ATTACK_SPECS if not selected_attack_ids or item.attack in selected_attack_ids)
    model_adapters = tuple(item for item in MAIN_MODELS if not selected_model_adapters or item in selected_model_adapters)

    if not args.skip_attacks:
        result["attack_matrix"] = _run_attack_matrix(
            api,
            out_dir=out_dir,
            timeout_seconds=int(args.attack_timeout),
            attack_specs=attack_specs,
        )
    if not args.skip_models:
        result["model_matrix"] = _run_model_matrix(
            api,
            out_dir=out_dir,
            timeout_seconds=int(args.model_timeout),
            model_adapters=model_adapters,
        )

    result["overall_passed"] = bool(
        overview["checks"]["attack_count_ok"]
        and overview["checks"]["model_count_ok"]
        and overview["checks"]["dataset_count_ok"]
        and overview["checks"]["main_models_present"]
        and overview["checks"]["backend_version_matches"]
        and overview["checks"]["deployment_target_matches"]
        and overview["checks"]["frontend_assets_match"]
        and overview["checks"]["frontend_dist_fresh"]
        and (args.skip_attacks or bool(result["attack_matrix"]["all_success"]))
        and (args.skip_models or bool(result["model_matrix"]["all_success"]))
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)
    return 0 if result["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
