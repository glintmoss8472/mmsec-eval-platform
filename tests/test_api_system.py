# 文件说明：该文件属于自动化测试，集中实现 test api system 相关逻辑。
from __future__ import annotations

import json
from pathlib import Path

import pytest

from api_test_utils import make_client


# 写出 `system 数据集 fixtures`，保证后续报告、页面或复现实验能读取。
def _write_system_dataset_fixtures(client, tmp_path: Path) -> None:
    store = client.app.state.store
    coco_root = tmp_path / "datasets" / "coco"
    (coco_root / "val2017").mkdir(parents=True)
    (coco_root / "annotations").mkdir(parents=True)
    (coco_root / "annotations" / "captions_val2017_subset.json").write_text(
        json.dumps({"annotations": [{} for _ in range(256)]}, ensure_ascii=False),
        encoding="utf-8",
    )
    flickr_root = tmp_path / "datasets" / "flickr30k"
    (flickr_root / "images").mkdir(parents=True)
    (flickr_root / "captions_index.jsonl").write_text(
        "\n".join(json.dumps({"image": f"{idx}.jpg", "caption": f"caption {idx}"}, ensure_ascii=False) for idx in range(16)),
        encoding="utf-8",
    )
    (flickr_root / "captions_index_single.jsonl").write_text(
        "\n".join(json.dumps({"image": f"slice_{idx}.jpg", "caption": f"slice caption {idx}"}, ensure_ascii=False) for idx in range(8)),
        encoding="utf-8",
    )
    mini_root = tmp_path / "datasets" / "mini_flickr"
    (mini_root / "images").mkdir(parents=True)
    (mini_root / "captions_index.jsonl").write_text(
        "\n".join(json.dumps({"image": f"mini_{idx}.jpg", "caption": f"mini {idx}"}, ensure_ascii=False) for idx in range(4)),
        encoding="utf-8",
    )
    store.upsert_dataset("coco_subset", str(coco_root), True, 256, "prepared via api")
    store.upsert_dataset("flickr30k", str(flickr_root), True, 16, "prepared via api")
    store.upsert_dataset("flickr1k", str(flickr_root), True, 8, "prepared via api")
    store.upsert_dataset("mini_flickr", str(mini_root), True, 4, "prepared via api (demo fixture)")


# 判断或归一 `assert 数据集 endpoint ready` 状态，让调用方可以稳定渲染能力和可用性。
def _assert_dataset_endpoint_ready(client) -> None:
    datasets_resp = client.get("/api/v1/datasets")
    assert datasets_resp.status_code == 200
    dataset_items = {item["name"]: item for item in datasets_resp.json()["items"]}
    for name in ("coco_subset", "flickr30k", "flickr1k", "mini_flickr"):
        assert dataset_items[name]["prepared"] is True
        assert dataset_items[name]["ready"] is True
        assert dataset_items[name]["ready_reason"] == ""


# 归类 `assert 系统总览 top level`，把连续分数或多条记录整理成稳定分组。
def _assert_overview_top_level(payload: dict) -> None:
    required = {
        "project_root", "adapters", "models", "attacks", "datasets", "paper_repositories",
        "validation_summary", "latest_formal_runs", "latest_primary_formal_runs",
        "latest_ablation_runs", "primary_formal_runs_source_path", "primary_formal_runs_source_kind",
        "live_runtime_note", "paper_result_environment_source_path", "paper_result_environment_note",
        "paper_result_environment", "build_identity", "model_coverage",
        "dataset_catalog", "dataset_catalog_formal_count", "dataset_catalog_total_count",
        "external_attack_status",
    }
    assert required.issubset(payload)
    assert len(payload["models"]) >= 10
    assert payload["model_total_count"] == len(payload["models"])
    formal_models = [item for item in payload["models"] if item.get("formal_eval") is not False]
    assert payload["supported_model_count"] == len(formal_models)
    assert all(item != "fixture_vlm" for item in payload["model_coverage"]["integrated"]["models"])
    assert all(item != "fixture_vlm" for item in payload["online_models"])
    assert payload["model_coverage"]["integrated"]["count"] == payload["supported_model_count"]
    assert payload["model_coverage"]["online"]["count"] == payload["online_model_count"]


# 判断或归一 `assert external 攻击 状态` 状态，让调用方可以稳定渲染能力和可用性。
def _assert_external_attack_status(payload: dict) -> None:
    statuses = payload["external_attack_status"]
    expected = {"vqa_visual_corruption", "xtransfer_uap", "foa_attack", "anyattack", "mpc_attack", "m_attack"}
    assert expected.issubset(statuses)
    for attack_id in expected:
        item = statuses[attack_id]
        assert {"repo", "checkpoint", "target", "config_path", "runnable", "messages"}.issubset(item)
        assert item["repo"]["status"] in {"ready", "missing", "not_required"}
        assert item["checkpoint"]["status"] in {"ready", "missing", "not_required"}
        assert item["target"]["status"] in {"ready", "missing", "not_required"}
    assert statuses["vqa_visual_corruption"]["repo"]["required"] is True
    assert statuses["vqa_visual_corruption"]["checkpoint"]["status"] == "not_required"
    assert statuses["anyattack"]["checkpoint"]["required"] is True
    assert statuses["foa_attack"]["target"]["required"] is True


# 执行 `assert 系统总览 数据集` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _assert_overview_datasets(payload: dict) -> None:
    assert len(payload["datasets"]) == payload["dataset_total_count"]
    assert payload["dataset_total_count"] == 4
    assert payload["formal_dataset_count"] == 3
    assert payload["dataset_catalog_formal_count"] == 6
    assert payload["dataset_catalog_total_count"] == 7
    assert {item["key"] for item in payload["datasets"]} == {"coco_subset", "flickr30k", "flickr1k", "mini_flickr"}
    assert all(item["ready"] is True for item in payload["datasets"])
    assert any(item["key"] == "mini_flickr" and item.get("tier") == "demo" for item in payload["datasets"])
    assert any(item["key"] == "mini_flickr" and item.get("tier") == "demo" for item in payload["dataset_catalog"])
    assert any(item["key"] == "vqa_v2_coco_val" and item.get("tier") == "generation" for item in payload["dataset_catalog"])
    assert any(item["key"] == "coco_caption_object_val" and item.get("tier") == "generation" for item in payload["dataset_catalog"])


# 执行 `assert 系统总览 environment` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _assert_overview_environment(payload: dict) -> None:
    assert payload["primary_formal_runs_source_path"].endswith("artifacts/paper_suite_20260418_final/paper_suite_analysis.json")
    assert payload["primary_formal_runs_source_kind"] == "canonical_thesis_suite"
    assert payload["paper_result_environment_source_path"].endswith("artifacts/paper_suite_20260418_final/environment_reference.json")
    assert payload["paper_result_environment"]["python_version"] == "3.8.10"
    assert payload["paper_result_environment"]["torch"]["version"] == "2.1.0+cu121"
    assert payload["paper_result_environment"]["torch"]["cuda_version"] == "12.1"
    assert payload["paper_result_environment_note"]
    assert payload["build_identity"]["runtime_transport"] == "host_python"
    assert payload["build_identity"]["runtime_context"] == "host"
    assert payload["validated_model_count"] == 0


# 执行 `assert primary 正式结果 运行记录` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _assert_primary_formal_runs(payload: dict) -> None:
    assert payload["primary_formal_runs_artifact_index_path"].endswith("artifacts/paper_suite_20260418_final/row_artifact_index.json")
    if not payload["latest_primary_formal_runs"]:
        return
    rows = payload["latest_primary_formal_runs"]
    primary_row = rows[0]
    assert primary_row["experiment_label"] and primary_row["evidence_group"] == "primary"
    assert primary_row["run_id"] == "20260418_045024_740869"
    assert primary_row["evidence_row_id"] == "paper_e1_advclip_coco"
    assert primary_row["artifact_index_path"] == "artifacts/paper_suite_20260418_final/row_artifact_index.json"
    assert primary_row["asr_attack"] == pytest.approx(0.046875)
    assert primary_row["metric_label"] == "首位攻击成功率（attack success rate at first rank，汇总）"
    assert primary_row["k_value"] == 1
    assert primary_row["retrieval_direction_scope"] == "图检文与文检图双向平均"
    assert primary_row["victim_aggregation"] == "3 个受测模型平均"
    assert primary_row["sample_pair_count"] == 32
    assert "attack success rate at first rank" in primary_row["metric_note"]
    assert {"E1 主实验", "E2 主实验"}.issubset({item["suite_label"] for item in rows})
    assert {"advclip", "tmm", "advedm", "advedm_plus"}.issubset({item["attack"] for item in rows})
    assert {"tmm", "advedm_plus"}.issubset({item["attack"] for item in rows if item.get("joint_execution_declared")})
    assert {"图像", "图文联合"}.issubset({item.get("attack_modality", "") for item in rows})


# 整理 `assert primary 产物 paths` 路径信息，把本地文件或产物引用转换成统一表示。
def _assert_primary_artifact_paths(payload: dict) -> None:
    if not payload["latest_primary_formal_runs"]:
        return
    primary_row = payload["latest_primary_formal_runs"][0]
    project_root = Path(payload["project_root"])
    for key in ("archived_summary_path", "archived_report_path", "portable_report_data_path", "portable_report_path", "artifact_index_path"):
        assert (project_root / primary_row[key]).exists()
    assert primary_row["source_summary_path"].endswith("/artifacts/runs/20260418_045024_740869/summary.json")
    assert primary_row["source_report_data_path"].endswith("/artifacts/runs/20260418_045024_740869/report_data.json")
    assert primary_row["source_report_path"].endswith("/artifacts/runs/20260418_045024_740869/report.html")
    assert primary_row["summary_path"].endswith("/artifacts/runs/20260418_045024_740869/report_data.json")
    assert primary_row["report_path"].endswith("/artifacts/runs/20260418_045024_740869/report.html")
    assert primary_row["portable_report_data_path"] == "artifacts/paper_suite_20260418_final/rows/paper_e1_advclip_coco/portable_report_data.json"
    assert primary_row["portable_report_path"] == "artifacts/paper_suite_20260418_final/rows/paper_e1_advclip_coco/portable_report.html"


# 执行 `assert 消融 运行记录` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _assert_ablation_runs(payload: dict) -> None:
    if not payload["latest_ablation_runs"]:
        return
    ablation_row = payload["latest_ablation_runs"][0]
    assert ablation_row["experiment_label"].startswith("E4")
    assert ablation_row["evidence_group"] == "ablation"
    assert ablation_row["evidence_row_id"]
    assert ablation_row["archived_summary_path"].startswith("artifacts/paper_suite_20260418_final/rows/")


# 组装 `assert compliance 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _assert_compliance_payload(cp: dict) -> None:
    assert {"checklist_semantics", "taskbook_items", "paper_coverage", "engineering_views"}.issubset(cp)
    assert "result_conformance" in cp
    assert "model_validation_ok" in cp["result_conformance"]
    routes = {item["route"] for item in cp["engineering_views"]["ui_pages"]}
    assert {
        "/",
        "/testing",
        "/jobs",
        "/analysis",
        "/cases",
        "/reports",
        "/reports/:runId",
        "/reports/:runId/cases/:sampleId",
    }.issubset(routes)
    assert "/glossary" not in routes
    assert cp["paper_coverage"]
    assert {item["reproduction_fidelity"] for item in cp["paper_coverage"]} == {"approx"}
    taskbook_items = {item["id"]: item for item in cp["taskbook_items"]}
    assert taskbook_items["req_1"]["status"] in {"ready", "partial"}
    for token in ("registered_attacks=", "observed_attack_coverage_count=", "observed_image_only_execution_report_runs=", "observed_joint_execution_report_runs=", "observed_joint_execution_attacks="):
        assert token in taskbook_items["req_1"]["evidence"]
    assert "three_stage_coverage_complete=" in taskbook_items["req_2"]["evidence"]
    assert taskbook_items["req_3"]["status"] == "ready"
    assert taskbook_items["req_4"]["status"] == "partial"
    assert "portable_formal_report_runs=" in taskbook_items["req_3"]["evidence"]


# 验证 `system 系统总览` 场景，防止相关行为在后续修改中退化。
def test_system_overview(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        _write_system_dataset_fixtures(client, tmp_path)
        _assert_dataset_endpoint_ready(client)
        r = client.get("/api/v1/system/overview")
        assert r.status_code == 200
        payload = r.json()
        _assert_overview_top_level(payload)
        _assert_external_attack_status(payload)
        _assert_overview_datasets(payload)
        _assert_overview_environment(payload)
        _assert_primary_formal_runs(payload)
        _assert_primary_artifact_paths(payload)
        _assert_ablation_runs(payload)
        c = client.get("/api/v1/system/compliance")
        assert c.status_code == 200
        _assert_compliance_payload(c.json())


# 验证 `数据集 endpoint splits prepared 来源 ready` 场景，防止相关行为在后续修改中退化。
def test_datasets_endpoint_splits_prepared_from_ready(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        store = client.app.state.store
        ready_root = tmp_path / "datasets" / "coco_ready"
        (ready_root / "val2017").mkdir(parents=True)
        (ready_root / "annotations").mkdir(parents=True)
        (ready_root / "annotations" / "captions_val2017_subset.json").write_text(
            json.dumps({"annotations": [{}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        broken_root = tmp_path / "datasets" / "flickr_broken"
        broken_root.mkdir(parents=True)

        store.upsert_dataset("coco_subset", str(ready_root), True, 1, "ready dataset")
        store.upsert_dataset("flickr30k", str(broken_root), True, 16, "missing files")

        r = client.get("/api/v1/datasets")
        assert r.status_code == 200
        items = {item["name"]: item for item in r.json()["items"]}
        assert items["coco_subset"]["prepared"] is True
        assert items["coco_subset"]["ready"] is True
        assert items["flickr30k"]["prepared"] is True
        assert items["flickr30k"]["ready"] is False
        assert "缺少图像目录 images" in items["flickr30k"]["ready_reason"]

        overview = client.get("/api/v1/system/overview")
        assert overview.status_code == 200
        payload = overview.json()
        assert {item["key"] for item in payload["datasets"]} == {"coco_subset"}
        assert payload["dataset_total_count"] == 1
        assert payload["formal_dataset_count"] == 1


# 验证 `模型 endpoint exposes 任务 capabilities and excludes fixture 来源 正式结果 tasks` 场景，防止相关行为在后续修改中退化。
def test_models_endpoint_exposes_task_capabilities_and_excludes_fixture_from_formal_tasks(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        items = {item["adapter"]: item for item in r.json()["items"]}
        assert "fixture_vlm" not in items
        assert items["clip_hf"]["task_capabilities"] == ["vlr"]
        assert {"vlr", "vqa", "caption"}.issubset(set(items["openai_qwen35_9b"]["task_capabilities"]))


# 验证 `unknown API 路径 returns JSON 404 instead of spa HTML` 场景，防止相关行为在后续修改中退化。
def test_unknown_api_path_returns_json_404_instead_of_spa_html(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        r = client.get("/api/v1/not-a-real-endpoint")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith("application/json")
        assert "API route not found" in r.json()["detail"]


# 验证 `生成式评测 JSONL 数据集 are registry ready` 场景，防止相关行为在后续修改中退化。
def test_generation_jsonl_datasets_are_registry_ready(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        store = client.app.state.store
        generation_root = tmp_path / "datasets" / "generation"
        generation_root.mkdir(parents=True)
        (generation_root / "vqa_v2_coco_val.jsonl").write_text(
            "\n".join(json.dumps({"id": f"vqa-{idx}", "image": "x.jpg", "question": "Q?", "answer": "yes"}, ensure_ascii=False) for idx in range(2)),
            encoding="utf-8",
        )
        store.upsert_dataset("vqa_v2_coco_val", str(generation_root), True, 2, "prepared real VQA JSONL")

        r = client.get("/api/v1/datasets")
        assert r.status_code == 200
        item = {row["name"]: row for row in r.json()["items"]}["vqa_v2_coco_val"]
        assert item["prepared"] is True
        assert item["ready"] is True
        assert item["ready_reason"] == ""
        assert item["item_count"] == 2

        overview = client.get("/api/v1/system/overview")
        assert overview.status_code == 200
        payload = overview.json()
        live = {row["key"]: row for row in payload["datasets"]}
        assert live["vqa_v2_coco_val"]["tier"] == "generation"
        assert live["vqa_v2_coco_val"]["source"] == "dataset_registry"
        assert payload["dataset_total_count"] == 1
        assert payload["formal_dataset_count"] == 1


# 验证 `任务 进度 exposes stage 本地 hard counts` 场景，防止相关行为在后续修改中退化。
def test_job_progress_exposes_stage_local_hard_counts(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        store = client.app.state.store
        job = store.create_job(
            job_type="run_vlr",
            config_path="configs/mvp.yaml",
            override={},
            benchmark_mode=False,
            payload={},
        )
        store.init_job_progress(job["id"], "run_vlr")
        store.set_job_running(job["id"])
        store.update_job_stage(
            job["id"],
            "victim_evaluation",
            "running",
            82.25,
            "正在评测攻击后输入在各受测模型上的表现：openai_gemma3_12b，已完成 32/256 对图文配对。",
        )

        r = client.get(f"/api/v1/jobs/{job['id']}/progress")
        assert r.status_code == 200
        payload = r.json()
        assert payload["progress_percent"] == 82.25
        assert payload["current_stage"] == "victim_evaluation"
        assert payload["current_stage_units_done"] == 32
        assert payload["current_stage_units_total"] == 256
        assert payload["current_stage_progress_percent"] == 12.5
        assert payload["current_stage_message"].startswith("正在评测攻击后输入")
        assert payload["progress_percent_semantics"].startswith("overall pipeline completion percent")


# 验证 `success 任务 进度 prefers completed stage and closes running 报告` 场景，防止相关行为在后续修改中退化。
def test_success_job_progress_prefers_completed_stage_and_closes_running_report(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        store = client.app.state.store
        job = store.create_job(
            job_type="run_caption",
            config_path="configs/bench/bootstrap_quick_caption.yaml",
            override={},
            benchmark_mode=False,
            payload={},
        )
        store.init_job_progress(job["id"], "run_caption")
        store.set_job_running(job["id"])
        store.update_job_stage(job["id"], "attack_execution", "success", 83, "已完成 1 / 1 条生成式样本")
        store.update_job_stage(job["id"], "result_aggregation", "success", 90, "正在汇总生成式评测结果。")
        store.update_job_stage(job["id"], "report_writing", "running", 97, "正在写入报告。")
        store.set_job_success(job["id"], run_id="run-caption-1")
        store.update_job_stage(job["id"], "completed", "success", 100, "任务执行完成，运行编号：run-caption-1")
        store.update_job_stage(job["id"], "report_writing", "running", 97, "历史遗留：报告阶段仍显示运行中。")

        r = client.get(f"/api/v1/jobs/{job['id']}/progress")
        assert r.status_code == 200
        payload = r.json()
        assert payload["status"] == "success"
        assert payload["current_stage"] == "completed"
        assert payload["current_stage_message"] == "任务执行完成，运行编号：run-caption-1"
        stages = {stage["stage_key"]: stage for stage in payload["stages"]}
        assert stages["report_writing"]["state"] == "success"
        assert stages["completed"]["state"] == "success"


# 整理 `paper acceptance rows` 行记录，把原始结果转换成列表接口和报告可消费的结构。
def _paper_acceptance_rows(*, strong_defense: bool) -> list[dict]:
    strong_gains = {
        "paper_e1_tmm_coco": 0.12,
        "paper_e1_advedm_plus_coco": 0.2,
        "paper_e2_tmm_flickr": 0.1,
        "paper_e2_advedm_plus_flickr": 0.11,
    }
    weak_gains = {
        "paper_e1_tmm_coco": 0.04,
        "paper_e1_advedm_plus_coco": 0.05,
        "paper_e2_tmm_flickr": 0.06,
        "paper_e2_advedm_plus_flickr": 0.07,
    }
    gain_map = strong_gains if strong_defense else weak_gains
    rows = [
        ("E0_smoke", "paper_e0_smoke", "advedm_plus", 0.0, 0.0, 0.0),
        ("E1_classic_coco", "paper_e1_advclip_coco", "advclip", 0.03, 0.0, 0.2),
        ("E1_classic_coco", "paper_e1_tmm_coco", "tmm", 0.6, gain_map["paper_e1_tmm_coco"], 0.55),
        ("E1_classic_coco", "paper_e1_advedm_coco", "advedm", 0.05, 0.0, 0.32),
        ("E1_classic_coco", "paper_e1_advedm_plus_coco", "advedm_plus", 0.5, gain_map["paper_e1_advedm_plus_coco"], 0.56),
        ("E2_classic_flickr", "paper_e2_advclip_flickr", "advclip", 0.03, 0.0, 0.15),
        ("E2_classic_flickr", "paper_e2_tmm_flickr", "tmm", 0.58, gain_map["paper_e2_tmm_flickr"], 0.54),
        ("E2_classic_flickr", "paper_e2_advedm_flickr", "advedm", 0.04, 0.0, 0.31),
        ("E2_classic_flickr", "paper_e2_advedm_plus_flickr", "advedm_plus", 0.34, gain_map["paper_e2_advedm_plus_flickr"], 0.42),
        ("E4_ablation", "paper_e4_advedm_plus_full", "advedm_plus", 0.42, 0.0, 0.32),
        ("E4_ablation", "paper_e4_advedm_plus_no_text", "advedm_plus", 0.16, 0.0, 0.61),
        ("E4_ablation", "paper_e4_advedm_plus_no_adaptive", "advedm_plus", 0.39, 0.0, 0.59),
        ("E4_ablation", "paper_e4_advedm_plus_no_fixation", "advedm_plus", 0.4, 0.0, 0.58),
    ]
    return [
        {
            "suite": suite,
            "id": row_id,
            "attack": attack,
            "asr_attack": asr_attack,
            "defense_gain": defense_gain,
            "risk_score": risk_score,
            "num_victim_failures": 0,
        }
        for suite, row_id, attack, asr_attack, defense_gain, risk_score in rows
    ]


# 组装 `模型 验证 fixture 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _model_validation_fixture_payload() -> dict:
    return {
        "passed": True,
        "dataset_name": "flickr1k",
        "attacks": ["fgsm", "advedm_plus"],
        "validated_models": [
            "clip_hf",
            "blip_itm",
            "vilt_itm",
            "openai_qwen35_9b",
            "openai_qwen3_vl",
            "openai_qwen25_vl",
            "openai_internvl35",
            "openai_minicpm_v",
            "openai_ovis25",
            "openai_gemma3_12b",
        ],
        "validated_model_count": 10,
        "criterion": {
            "benchmark_attacks": ["fgsm", "advedm_plus"],
            "max_pairs": 256,
            "minimum_qualifying_attack_count_per_model": 1,
            "minimum_attack_asr_any": 0.02,
            "minimum_attack_drop_r1_any": 0.02,
        },
        "rows": [
            {"attack": "fgsm", "asr_attack": 0.41, "defense_gain": 0.25},
            {"attack": "advedm_plus", "asr_attack": 0.34, "defense_gain": 0.22},
            {"attack": "advedm_plus", "asr_attack": 0.16, "defense_gain": 0.15},
        ],
    }


# 写出 `result conformance fixture`，保证后续报告、页面或复现实验能读取。
def _write_result_conformance_fixture(
    tmp_project_name: str,
    *,
    paper_suite_name: str,
    validation_name: str,
    strong_defense: bool,
) -> tuple[Path, Path]:
    project_root = Path.cwd()
    thresholds = json.loads((project_root / "artifacts" / "paper_acceptance_thresholds.json").read_text(encoding="utf-8"))
    tmp_project = project_root / "tmp" / tmp_project_name
    artifacts_dir = tmp_project / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "paper_acceptance_thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paper_suite_dir = artifacts_dir / paper_suite_name
    paper_suite_dir.mkdir(parents=True, exist_ok=True)
    (paper_suite_dir / "status.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
    analysis = {"phase_count": 4, "row_count": 13, "rows": _paper_acceptance_rows(strong_defense=strong_defense)}
    (paper_suite_dir / "paper_suite_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_dir = artifacts_dir / validation_name
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "status.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
    (validation_dir / "summary.json").write_text(
        json.dumps(_model_validation_fixture_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return tmp_project, artifacts_dir


# 验证 `result conformance contract uses 验证 rows 所属 防御 and 模型 验证` 场景，防止相关行为在后续修改中退化。
def test_result_conformance_contract_uses_validation_rows_for_defense_and_model_validation():
    from mmsec_api.services import system_overview as system_overview_module

    tmp_project, artifacts_dir = _write_result_conformance_fixture(
        "test_result_conformance_project",
        paper_suite_name="paper_suite_20990101_final",
        validation_name="model_validation_20990101_final",
        strong_defense=True,
    )

    result = system_overview_module._result_conformance_v2(tmp_project, artifacts_dir)

    assert result["e1_ok"] is True
    assert result["e2_ok"] is True
    assert result["e4_ok"] is True
    assert result["defense_ok"] is True
    assert result["adaptive_ok"] is True
    assert result["fixation_ok"] is True
    assert result["model_validation_ok"] is True
    assert result["passed"] is True
    assert all("risk gain" not in item for item in result["caveats"])


# 验证 `result conformance rejects weak 防御 gain even when rows are effective` 场景，防止相关行为在后续修改中退化。
def test_result_conformance_rejects_weak_defense_gain_even_when_rows_are_effective():
    from mmsec_api.services import system_overview as system_overview_module

    tmp_project, artifacts_dir = _write_result_conformance_fixture(
        "test_result_conformance_weak_defense_project",
        paper_suite_name="paper_suite_20990102_final",
        validation_name="model_validation_20990102_final",
        strong_defense=False,
    )

    result = system_overview_module._result_conformance_v2(tmp_project, artifacts_dir)

    assert result["e1_ok"] is True
    assert result["e2_ok"] is True
    assert result["e4_ok"] is True
    assert result["defense_ok"] is False
    assert result["model_validation_ok"] is True
    assert result["passed"] is False
    assert "Defense recovery is still below the baseline acceptance threshold" in result["caveats"]


# 执行 `补丁 taskbook contract dependencies` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _patch_taskbook_contract_dependencies(system_overview_module, monkeypatch) -> None:
    monkeypatch.setattr(
        system_overview_module,
        "list_plugins",
        lambda kind: [
            "advclip",
            "tmm",
            "advedm",
            "advedm_plus",
            "fgsm",
            "bim",
            "pgd",
            "mifgsm",
            "nifgsm",
            "difgsm",
            "tifgsm",
            "dtmifgsm",
            "vmifgsm",
            "vnifgsm",
            "cw",
        ]
        if kind == "attack"
        else [],
    )
    monkeypatch.setattr(system_overview_module, "_frontend_ready", lambda *_: True)
    monkeypatch.setattr(system_overview_module, "_core_routes_ready", lambda *_: True)


# 写出 `taskbook 运行记录 fixture`，保证后续报告、页面或复现实验能读取。
def _write_taskbook_run_fixture(artifacts_dir: Path, *, run_id: str, attack: str, scope: str, text_changed_ratio: float) -> Path:
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    is_joint = scope == "joint"
    summary = {
        "run_id": run_id.replace("20260419_000000_", "20260419_"),
        "attack": attack,
        "dataset_name": "flickr1k",
        "model_adapter": "clip_hf",
        "asr": 0.41 if is_joint else 0.25,
        "asr_attack": 0.41 if is_joint else 0.25,
        "asr_defended": 0.12 if is_joint else 0.1,
        "attack_debug": {
            "scope": scope,
            "need_image": True,
            "need_text": is_joint,
            "num_images_attacked": 1,
            "num_texts_attacked": 1 if is_joint else 0,
            "text_changed_ratio": text_changed_ratio,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report_data.json").write_text(
        json.dumps({"stage_metrics": {"clean": {"r1": 0.9}, "attacked": {"r1": 0.6}, "defended_attack": {"r1": 0.7}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "report.html").write_text(f"<html><body>{scope}</body></html>", encoding="utf-8")
    return run_dir


# 写出 `taskbook paper suite fixture`，保证后续报告、页面或复现实验能读取。
def _write_taskbook_paper_suite_fixture(artifacts_dir: Path) -> None:
    paper_suite_dir = artifacts_dir / "paper_suite_20990101_final"
    paper_suite_dir.mkdir(parents=True, exist_ok=True)
    (paper_suite_dir / "status.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
    rows = [
        {
            "suite": "E1_classic_coco", "run_id": "20260419_000000_demo", "attack": "fgsm",
            "eval_scope": "image", "dataset_name": "flickr1k", "benchmark_tag": "paper_demo_row",
            "victim_model_adapters": ["clip_hf"], "asr_attack": 0.25, "asr_defended": 0.1,
            "defense_gain": 0.15, "risk_score": 0.3, "avg_l2": 1.0,
        },
        {
            "suite": "E2_classic_flickr", "run_id": "20260419_000001_joint", "attack": "tmm",
            "eval_scope": "joint", "dataset_name": "flickr1k", "benchmark_tag": "paper_demo_joint_row",
            "victim_model_adapters": ["clip_hf"], "asr_attack": 0.41, "asr_defended": 0.12,
            "defense_gain": 0.29, "risk_score": 0.52, "avg_l2": 1.2,
        },
    ]
    (paper_suite_dir / "paper_suite_analysis.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


# 写出 `taskbook 样本 产物`，保证后续报告、页面或复现实验能读取。
def _write_taskbook_sample_artifacts(artifacts_dir: Path, run_dir: Path) -> None:
    (artifacts_dir / "advclip_patch_registry.json").write_text(
        json.dumps({"version": 1, "entries": {"demo_patch": {"path": "artifacts/patches/demo.png"}}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "cases_index.jsonl").write_text(json.dumps({"case_id": "seed-000"}, ensure_ascii=False) + "\n", encoding="utf-8")
    case_dir = run_dir / "cases" / "seed-000"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_bundle.json").write_text(json.dumps({"case_id": "seed-000"}, ensure_ascii=False, indent=2), encoding="utf-8")
    debug_dir = run_dir / "attack_debug" / "seed-000"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "debug.json").write_text(json.dumps({"ok": True}, ensure_ascii=False, indent=2), encoding="utf-8")


# 写出 `taskbook 验证 fixture`，保证后续报告、页面或复现实验能读取。
def _write_taskbook_validation_fixture(artifacts_dir: Path) -> None:
    validation_dir = artifacts_dir / "model_validation_20990101_final"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "status.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
    payload = {
        "passed": True,
        "dataset_name": "flickr1k",
        "attacks": ["fgsm", "advedm_plus"],
        "validated_model_count": 10,
        "criterion": {
            "dataset_name": "flickr1k",
            "benchmark_attacks": ["fgsm", "advedm_plus"],
            "max_pairs": 256,
            "minimum_attack_asr_any": 0.02,
            "minimum_attack_drop_r1_any": 0.02,
        },
    }
    (validation_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 组装 `样本 management ready 载荷`，把分散字段整理成后端任务或风险评分使用的载荷。
def _sample_management_ready_payload() -> dict:
    return {
        "cases_index_runs": 1,
        "case_bundles": 1,
        "attack_debug_cases": 1,
        "patch_registry_entries": 1,
    }


# 汇总 `taskbook 产物 摘要`，从运行记录和指标中提炼页面展示所需的分析结果。
def _taskbook_artifact_summary(**overrides) -> dict:
    payload = {
        "archived_row_evidence_runs": 2,
        "archived_metric_ready_runs": 2,
        "formal_report_runs": 2,
        "formal_metric_ready_runs": 2,
        "source_formal_report_runs": 2,
        "source_formal_metric_ready_runs": 2,
        "portable_formal_report_runs": 2,
        "portable_formal_metric_ready_runs": 2,
        "three_stage_runs": 2,
        "distinct_attacks": 2,
        "distinct_victim_models": 3,
        "image_only_definition_rows": 1,
        "joint_definition_rows": 1,
        "image_only_definition_attacks": ["advclip"],
        "joint_definition_attacks": ["tmm"],
    }
    payload.update(overrides)
    return payload


# 汇总 `taskbook observed 摘要`，从运行记录和指标中提炼页面展示所需的分析结果。
def _taskbook_observed_summary(**overrides) -> dict:
    payload = {
        "image_only_execution_report_runs": 1,
        "joint_execution_report_runs": 1,
        "image_only_execution_attacks": ["advclip"],
        "joint_execution_attacks": ["tmm"],
    }
    payload.update(overrides)
    return payload


# 执行 `补丁 taskbook gap dependencies` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _patch_taskbook_gap_dependencies(
    system_overview_module,
    monkeypatch,
    artifacts_dir: Path,
    *,
    formal_runs: list[dict],
    artifact_summary: dict,
    observed_summary: dict,
) -> None:
    _patch_taskbook_contract_dependencies(system_overview_module, monkeypatch)
    monkeypatch.setattr(system_overview_module, "_sample_management_artifacts", lambda _: _sample_management_ready_payload())
    monkeypatch.setattr(system_overview_module, "_validation_matrix_ready", lambda *_: (True, 10))
    monkeypatch.setattr(
        system_overview_module,
        "_latest_model_validation_summary",
        lambda *_: (artifacts_dir / "summary.json", {"validated_model_count": 10}),
    )
    monkeypatch.setattr(system_overview_module, "_latest_formal_runs", lambda *_args, **_kwargs: formal_runs)
    monkeypatch.setattr(system_overview_module, "_primary_formal_runs", lambda *_args, **_kwargs: formal_runs)
    monkeypatch.setattr(system_overview_module, "_official_formal_artifacts_summary", lambda *_args, **_kwargs: artifact_summary)
    monkeypatch.setattr(system_overview_module, "_observed_execution_summary", lambda *_args, **_kwargs: observed_summary)
    monkeypatch.setattr(system_overview_module, "_exists", lambda *_: True)


# 验证 `taskbook items contract bind to 产物 shapes and 验证` 场景，防止相关行为在后续修改中退化。
def test_taskbook_items_contract_bind_to_artifact_shapes_and_validation(tmp_path: Path, monkeypatch):
    from mmsec_api.services import system_overview as system_overview_module

    project_root = Path.cwd()
    artifacts_dir = tmp_path / "artifacts"
    _patch_taskbook_contract_dependencies(system_overview_module, monkeypatch)
    run_dir = artifacts_dir / "runs" / "20260419_000000_demo"
    _write_taskbook_run_fixture(artifacts_dir, run_id="20260419_000000_demo", attack="fgsm", scope="image", text_changed_ratio=0.0)
    _write_taskbook_run_fixture(artifacts_dir, run_id="20260419_000001_joint", attack="tmm", scope="joint", text_changed_ratio=0.25)
    _write_taskbook_paper_suite_fixture(artifacts_dir)

    items_before_validation = {item["id"]: item for item in system_overview_module._taskbook_items(project_root, artifacts_dir, core_api_ready=True)}
    assert items_before_validation["req_1"]["status"] == "partial"
    assert items_before_validation["req_2"]["status"] == "ready"
    assert items_before_validation["req_4"]["status"] == "partial"
    assert items_before_validation["req_5"]["status"] == "partial"

    _write_taskbook_sample_artifacts(artifacts_dir, run_dir)

    items_with_sample_artifacts = {item["id"]: item for item in system_overview_module._taskbook_items(project_root, artifacts_dir, core_api_ready=True)}
    assert items_with_sample_artifacts["req_1"]["status"] == "ready"
    assert "registered_attacks=" in items_with_sample_artifacts["req_1"]["evidence"]
    assert "observed_attack_coverage_count=" in items_with_sample_artifacts["req_1"]["evidence"]
    assert "case_bundles=1" in items_with_sample_artifacts["req_1"]["evidence"]
    assert "observed_joint_execution_attacks=" in items_with_sample_artifacts["req_1"]["evidence"]
    assert "tmm" in items_with_sample_artifacts["req_1"]["evidence"]

    _write_taskbook_validation_fixture(artifacts_dir)

    items_after_validation = {item["id"]: item for item in system_overview_module._taskbook_items(project_root, artifacts_dir, core_api_ready=True)}
    assert items_after_validation["req_4"]["status"] == "ready"
    assert items_after_validation["req_5"]["status"] == "ready"


# 验证 `primary paper suite analysis prefers canonical thesis suite over newer completed 目录` 场景，防止相关行为在后续修改中退化。
def test_primary_paper_suite_analysis_prefers_canonical_thesis_suite_over_newer_completed_dir(tmp_path: Path):
    from mmsec_api.services import system_overview as system_overview_module

    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    canonical_dir = artifacts_dir / "paper_suite_20260418_final"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / "paper_suite_analysis.json"
    canonical_path.write_text(
        json.dumps({"rows": [{"id": "canonical_row"}]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest_dir = artifacts_dir / "paper_suite_20990101_final"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "status.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
    (latest_dir / "paper_suite_analysis.json").write_text(
        json.dumps({"rows": [{"id": "latest_row"}]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    path, analysis, source_kind = system_overview_module._primary_paper_suite_analysis(project_root, artifacts_dir)

    assert path == canonical_path.resolve()
    assert analysis["rows"][0]["id"] == "canonical_row"
    assert source_kind == "canonical_thesis_suite"


# 验证 `frozen 证据 pack manifest declares completed subset boundary` 场景，防止相关行为在后续修改中退化。
def test_frozen_evidence_pack_manifest_declares_completed_subset_boundary():
    project_root = Path.cwd()
    manifest = json.loads(
        (project_root / "artifacts" / "defense_evidence_pack_20260419" / "manifest.json").read_text(encoding="utf-8")
    )

    thesis_basis = manifest["thesis_basis"]
    provenance = thesis_basis["provenance"]
    assert thesis_basis["basis_type"] == "frozen_analysis_packet"
    assert provenance["core_quantitative_row_count"] == 16
    assert provenance["audit_only_row_count"] == 3
    assert provenance["retained_local_run_count"] == 0
    assert "successful surviving row-level records" in provenance["provenance_note"]


# 验证 `最新 正式结果 运行记录 excludes demo and min verify 运行记录` 场景，防止相关行为在后续修改中退化。
def test_latest_formal_runs_excludes_demo_and_min_verify_runs(tmp_path: Path):
    from mmsec_api.services import system_overview as system_overview_module

    project_root = Path.cwd()
    artifacts_dir = tmp_path / "artifacts"
    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        ("20260419_000001_formal", "coco_subset", "paper_e1_advclip_coco64", "20260419_formal"),
        ("20260419_000002_demo", "mini_flickr", "demo_bundle", "20260419_demo"),
        ("20260419_000003_verify", "flickr1k", "validation_flickr1k_256", "fgsm_min_verify_3"),
        ("20260419_000004_quick", "flickr30k", "seed_bootstrap_vlr", ""),
    ]
    for run_id, dataset_name, benchmark_tag, experiment_id in run_specs:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "attack": "fgsm",
                    "dataset_name": dataset_name,
                    "benchmark_tag": benchmark_tag,
                    "experiment_id": experiment_id,
                    "model_adapter": "clip_hf",
                    "asr": 0.2,
                    "asr_attack": 0.2,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    paper_suite_dir = artifacts_dir / "paper_suite_20990101_final"
    paper_suite_dir.mkdir(parents=True, exist_ok=True)
    (paper_suite_dir / "status.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
    (paper_suite_dir / "paper_suite_analysis.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "suite": "E1_classic_coco",
                        "run_id": "20260419_000001_formal",
                        "attack": "fgsm",
                        "dataset_name": "coco_subset",
                        "benchmark_tag": "paper_e1_advclip_coco64",
                        "victim_model_adapters": ["clip_hf", "blip_itm", "vilt_itm"],
                        "asr_attack": 0.2,
                        "asr_defended": 0.1,
                        "defense_gain": 0.1,
                        "risk_score": 0.3,
                        "avg_l2": 1.0,
                    },
                    {
                        "suite": "E0_smoke",
                        "run_id": "20260419_000003_verify",
                        "attack": "fgsm",
                        "dataset_name": "flickr1k",
                        "benchmark_tag": "validation_flickr1k_256",
                        "victim_model_adapters": ["clip_hf"],
                        "asr_attack": 0.2,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = system_overview_module._latest_formal_runs(project_root, artifacts_dir, limit=10)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "20260419_000001_formal"
    assert rows[0]["dataset_name"] == "coco_subset"
    assert rows[0]["official_result"] is True


# 验证 `最新 正式结果 运行记录 keep 实验 tiers separate` 场景，防止相关行为在后续修改中退化。
def test_latest_formal_runs_keep_experiment_tiers_separate(tmp_path: Path):
    from mmsec_api.services import system_overview as system_overview_module

    project_root = Path.cwd()
    artifacts_dir = tmp_path / "artifacts"
    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        ("20260419_100001_e1", "coco_subset", "paper_e1_advclip_coco64", "paper_e1_advclip_coco", "advclip"),
        ("20260419_100002_e4", "coco_subset", "paper_e4_advedm_plus_full_coco24", "paper_e4_advedm_plus_full", "advedm_plus"),
    ]
    for run_id, dataset_name, benchmark_tag, experiment_id, attack in run_specs:
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "attack": attack,
                    "dataset_name": dataset_name,
                    "benchmark_tag": benchmark_tag,
                    "experiment_id": experiment_id,
                    "model_adapter": "clip_hf",
                    "surrogate_model_adapter": "clip_hf",
                    "victim_model_adapters": ["clip_hf", "blip_itm", "vilt_itm"],
                    "asr_attack": 0.5,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    paper_suite_dir = artifacts_dir / "paper_suite_20990101_final"
    paper_suite_dir.mkdir(parents=True, exist_ok=True)
    (paper_suite_dir / "status.json").write_text(json.dumps({"status": "completed"}, ensure_ascii=False), encoding="utf-8")
    (paper_suite_dir / "paper_suite_analysis.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": "paper_e1_advclip_coco",
                        "suite": "E1_classic_coco",
                        "run_id": "20260419_100001_e1",
                        "attack": "advclip",
                        "dataset_name": "coco_subset",
                        "benchmark_tag": "paper_e1_advclip_coco64",
                        "victim_model_adapters": ["clip_hf", "blip_itm", "vilt_itm"],
                        "asr_attack": 0.2,
                    },
                    {
                        "id": "paper_e4_advedm_plus_full",
                        "suite": "E4_ablation",
                        "run_id": "20260419_100002_e4",
                        "attack": "advedm_plus",
                        "dataset_name": "coco_subset",
                        "benchmark_tag": "paper_e4_advedm_plus_full_coco24",
                        "victim_model_adapters": ["clip_hf", "blip_itm", "vilt_itm"],
                        "asr_attack": 0.6667,
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = system_overview_module._latest_formal_runs(project_root, artifacts_dir, limit=10)
    primary_rows, ablation_rows = system_overview_module._split_formal_runs(rows, primary_limit=10, ablation_limit=10)

    assert [row["experiment_label"] for row in rows] == ["E1 主实验", "E4 消融 / 完整版本"]
    assert primary_rows[0]["evidence_group"] == "primary"
    assert primary_rows[0]["suite_label"] == "E1 主实验"
    assert ablation_rows[0]["evidence_group"] == "ablation"
    assert ablation_rows[0]["experiment_label"] == "E4 消融 / 完整版本"


# 验证 `taskbook req 1 requires official 联合 execution 证据` 场景，防止相关行为在后续修改中退化。
def test_taskbook_req_1_requires_official_joint_execution_evidence(monkeypatch):
    from mmsec_api.services import system_overview as system_overview_module

    project_root = Path.cwd()
    artifacts_dir = project_root / "tmp" / "test_taskbook_joint_execution_gap"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _patch_taskbook_gap_dependencies(
        system_overview_module,
        monkeypatch,
        artifacts_dir,
        formal_runs=[{"attack": "advclip"}],
        artifact_summary=_taskbook_artifact_summary(
            formal_report_runs=0,
            formal_metric_ready_runs=0,
            source_formal_report_runs=0,
            source_formal_metric_ready_runs=0,
            portable_formal_report_runs=0,
            portable_formal_metric_ready_runs=0,
            distinct_victim_models=2,
            image_only_definition_rows=2,
            image_only_definition_attacks=["advclip", "advedm"],
        ),
        observed_summary=_taskbook_observed_summary(
            image_only_execution_report_runs=0,
            joint_execution_report_runs=0,
            image_only_execution_attacks=[],
            joint_execution_attacks=[],
        ),
    )

    items = {
        item["id"]: item
        for item in system_overview_module._taskbook_items(project_root, artifacts_dir, core_api_ready=True)
    }

    assert items["req_1"]["status"] == "partial"
    assert "registered_attacks=" in items["req_1"]["evidence"]
    assert "observed_attack_coverage_count=0" in items["req_1"]["evidence"]
    assert "observed_joint_execution_report_runs=0" in items["req_1"]["evidence"]
    assert "observed_joint_execution_attacks=none" in items["req_1"]["evidence"]
    assert "image-only and image-text joint scenarios required by the taskbook" in items["req_1"]["gap"]


# 验证 `observed execution 摘要 skips legacy 运行记录 移除 攻击 调试` 场景，防止相关行为在后续修改中退化。
def test_observed_execution_summary_skips_legacy_runs_without_attack_debug(tmp_path: Path):
    from mmsec_api.services import system_overview as system_overview_module

    artifacts_dir = tmp_path / "artifacts"
    run_dir = artifacts_dir / "runs" / "20260422_legacy"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "20260422_legacy",
                "attack": "advedm",
                "dataset_name": "coco_subset",
                "asr_attack": 0.1,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "report.html").write_text("<html><body>legacy</body></html>", encoding="utf-8")

    summary = system_overview_module._observed_execution_summary(artifacts_dir)

    assert summary["image_only_execution_report_runs"] == 0
    assert summary["joint_execution_report_runs"] == 0
    assert summary["image_only_execution_attacks"] == []
    assert summary["joint_execution_attacks"] == []


# 验证 `taskbook req 2 requires three stage coverage across official 正式结果 运行记录` 场景，防止相关行为在后续修改中退化。
def test_taskbook_req_2_requires_three_stage_coverage_across_official_formal_runs(monkeypatch):
    from mmsec_api.services import system_overview as system_overview_module

    project_root = Path.cwd()
    artifacts_dir = project_root / "tmp" / "test_taskbook_req2_three_stage_gap"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    formal_runs = [
        {"attack": "advclip", "run_id": "formal_1"},
        {"attack": "tmm", "run_id": "formal_2"},
    ]
    _patch_taskbook_gap_dependencies(
        system_overview_module,
        monkeypatch,
        artifacts_dir,
        formal_runs=formal_runs,
        artifact_summary=_taskbook_artifact_summary(three_stage_runs=1),
        observed_summary=_taskbook_observed_summary(),
    )

    items = {
        item["id"]: item
        for item in system_overview_module._taskbook_items(project_root, artifacts_dir, core_api_ready=True)
    }

    assert items["req_2"]["status"] == "partial"
    assert "official_formal_runs=2" in items["req_2"]["evidence"]
    assert "official_three_stage_runs=1" in items["req_2"]["evidence"]
    assert "three_stage_coverage_complete=False" in items["req_2"]["evidence"]
    assert "not only representative samples" in items["req_2"]["gap"]
