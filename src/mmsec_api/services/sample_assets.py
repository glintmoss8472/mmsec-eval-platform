# 文件说明：该文件属于后端业务服务，集中实现 sample assets 相关逻辑。
from __future__ import annotations

from pathlib import Path
from typing import Any

from mmsec_api.services.run_reader import read_json
from mmsec_api.store.sqlite import SQLiteStore




# 定位 `运行记录 目录`，把配置值或请求上下文转换成实际文件系统路径。
def _run_dir(run_id: str, artifacts_dir: str = "artifacts") -> Path:
    return Path(artifacts_dir) / "runs" / run_id


# 转换 `float` 输入，在类型不匹配时回退到安全默认值。
def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)

# 确认 `record` 是字典记录，避免后续字段读取直接接触异常类型。
def _record(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# 规范化 `文本` 字段，把空值和非字符串输入转换为稳定文本。
def _text(value: object) -> str:
    return str(value or "").strip()


# 执行 `asset 作用范围` 辅助逻辑，保持后端业务服务中的输入处理和结果输出一致。
def _asset_scope(task_kind: str, attack: str, eval_scope: str) -> str:
    attack_key = attack.lower()
    if eval_scope.lower() == "joint" or attack_key in {"tmm", "advedm_plus"}:
        return "图文联合扰动"
    if task_kind in {"vqa", "caption"}:
        return "图像扰动"
    if "text" in attack_key:
        return "文本扰动"
    return "图像扰动"


# 判断或归一 `reuse state` 状态，让调用方可以稳定渲染能力和可用性。
def _reuse_state(artifact_status: str, clean_ref: str, adv_ref: str) -> tuple[str, str]:
    if clean_ref and adv_ref:
        return "ready", "原始图像与对抗图像均已保存，可纳入样本集复用。"
    if artifact_status == "complete":
        return "ready", "证据清单完整，可进入测评调用。"
    if artifact_status == "summary_only":
        return "summary_only", "历史运行仅保留指标或文本证据，建议重新生成后再纳入正式样本集。"
    return "legacy", "历史运行证据不完整，默认只作为复盘线索展示。"


# 整理 `证据包 refs` 路径信息，把本地文件或产物引用转换成统一表示。
def _bundle_refs(run_root: Path, sample_id: str) -> tuple[dict[str, Any], dict[str, str]]:
    bundle = read_json(run_root / "cases" / sample_id / "case_bundle.json", {})
    if not isinstance(bundle, dict):
        bundle = {}
    refs = _record(bundle.get("artifact_refs"))
    return bundle, {str(k): _text(v) for k, v in refs.items()}


# 执行 `asset 来源 案例` 辅助逻辑，保持后端业务服务中的输入处理和结果输出一致。
def asset_from_case(run: dict[str, object], row: dict[str, object], artifacts_dir: str) -> dict[str, object]:
    run_id = _text(row.get("run_id") or run.get("run_id"))
    sample_id = _text(row.get("sample_id"))
    run_root = _run_dir(run_id, artifacts_dir)
    bundle, refs = _bundle_refs(run_root, sample_id)
    sample = _record(bundle.get("sample"))
    adversarial = _record(bundle.get("adversarial"))
    adv_meta = _record(adversarial.get("metadata"))
    metrics = _record(bundle.get("metrics"))
    task_kind = _text(row.get("task_kind") or bundle.get("task_kind") or run.get("task_kind"))
    attack = _text(row.get("attack") or adv_meta.get("attack") or adv_meta.get("attack_name") or run.get("attack"))
    dataset = _text(row.get("benchmark_tag") or row.get("dataset_name") or bundle.get("dataset_tag") or run.get("benchmark_tag") or run.get("dataset_name"))
    model = _text(row.get("model_adapter") or bundle.get("model_tag") or run.get("model_adapter"))
    artifact_status = _text(row.get("artifact_status") or bundle.get("artifact_status") or "summary_only")
    clean_ref = refs.get("clean_image", "")
    adv_ref = refs.get("adv_image", "") or refs.get("attack_visualization", "")
    reusable_status, reusable_note = _reuse_state(artifact_status, clean_ref, adv_ref)
    eval_scope = _text(run.get("eval_scope") or bundle.get("eval_scope") or adv_meta.get("attack_scope"))
    clean_input = _record(_record(bundle.get("inputs")).get("clean"))
    text_value = _text(row.get("text") or sample.get("text") or clean_input.get("text"))
    asset_id = f"{run_id}::{sample_id}"
    return {
        "asset_id": asset_id,
        "variant_id": f"{asset_id}::{attack or 'attack'}",
        "source_run_id": run_id,
        "source_case_id": sample_id,
        "run_id": run_id,
        "sample_id": sample_id,
        "task_kind": task_kind,
        "dataset_name": _text(row.get("dataset_name") or run.get("dataset_name")),
        "benchmark_tag": dataset,
        "model_adapter": model,
        "attack": attack,
        "attack_scope": _asset_scope(task_kind, attack, eval_scope),
        "source_text": text_value,
        "target_text": _text(sample.get("target_text") or row.get("target_object") or row.get("gt_image_id")),
        "clean_image_ref": clean_ref,
        "adv_image_ref": adv_ref,
        "artifact_status": artifact_status,
        "reusable_status": reusable_status,
        "reusable_note": reusable_note,
        "judge_success": bool(row.get("judge_success")) if "judge_success" in row else bool(_record(bundle.get("judge")).get("success")),
        "risk_level": _text(row.get("risk_level") or run.get("risk_level")),
        "risk_score": _to_float(row.get("risk_score") or run.get("risk_score"), 0.0),
        "perturbation_l2": _to_float(row.get("perturbation_l2") or metrics.get("perturbation_l2") or adversarial.get("perturbation_l2"), 0.0),
        "perturbation_linf": _to_float(row.get("perturbation_linf") or metrics.get("perturbation_linf") or adversarial.get("perturbation_linf"), 0.0),
        "semantic_score": _to_float(metrics.get("semantic_preservation_rate") or metrics.get("caption_text_similarity"), 0.0),
        "created_at": _text(row.get("created_at") or run.get("created_at") or run_id),
        "metadata": {
            "source_run_id": run_id,
            "source_case_id": sample_id,
            "source_report_url": f"/reports/{run_id}",
            "source_case_url": f"/reports/{run_id}/cases/{sample_id}",
            "asset_scope": _asset_scope(task_kind, attack, eval_scope),
        },
    }


# 执行 `skip asset 评测 运行记录` 辅助逻辑，保持后端业务服务中的输入处理和结果输出一致。
def _skip_asset_eval_run(run_id: str, artifacts_dir: str) -> bool:
    summary = read_json(_run_dir(run_id, artifacts_dir) / "summary.json", {})
    return isinstance(summary, dict) and bool(summary.get("asset_evaluation_mode"))


# 收集 `资产 来源 运行记录`，把分散产物整理成统一列表。
def collect_assets_from_runs(artifacts_dir: str, store: SQLiteStore, run_ids: list[str] | None = None) -> list[dict[str, object]]:
    from mmsec_api.routes.runs import _case_rows_for_run, _decorated_rows_all

    wanted = {str(item).strip() for item in (run_ids or []) if str(item).strip()}
    assets: list[dict[str, object]] = []
    seen: set[str] = set()
    for run in _decorated_rows_all(artifacts_dir, store):
        run_id = _text(run.get("run_id"))
        if wanted and run_id not in wanted:
            continue
        if _skip_asset_eval_run(run_id, artifacts_dir):
            continue
        for row in _case_rows_for_run(run, artifacts_dir):
            sample_id = _text(row.get("sample_id"))
            if not sample_id or not run_id:
                continue
            key = f"{run_id}::{sample_id}"
            if key in seen:
                continue
            seen.add(key)
            assets.append(asset_from_case(run, row, artifacts_dir))
    return assets


# 同步 `样本 资产 来源 运行记录`，让数据库状态和产物目录保持一致。
def sync_sample_assets_from_runs(artifacts_dir: str, store: SQLiteStore, run_ids: list[str] | None = None) -> int:
    assets = collect_assets_from_runs(artifacts_dir, store, run_ids=run_ids)
    return store.upsert_sample_assets(assets)
