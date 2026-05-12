# 文件说明：该文件属于自动化测试，集中实现 test api runs null compat 相关逻辑。
from __future__ import annotations

from pathlib import Path

from api_test_utils import make_client


# 验证 `运行记录 list tolerates null cache fields` 场景，防止相关行为在后续修改中退化。
def test_runs_list_tolerates_null_cache_fields(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        store = client.app.state.store
        run_dir = tmp_path / "artifacts" / "runs" / "r_null_1"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text('{"run_id":"r_null_1"}', encoding="utf-8")
        with store._connect() as conn:  # noqa: SLF001 - test-only direct DB seed
            conn.execute(
                """
                INSERT INTO runs_cache(
                    run_id, created_at, task_kind, dataset_name, benchmark_tag, attack, mode, defense, experiment_id,
                    model_adapter, asr, asr_attack, asr_defended, defense_gain, avg_l2, path
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "r_null_1",
                    "2026-02-16T00:00:00+00:00",
                    None,
                    "toy_shapes",
                    None,
                    "advclip",
                    None,
                    None,
                    None,
                    "clip_hf",
                    None,
                    None,
                    None,
                    None,
                    None,
                    str(run_dir),
                ),
            )
            conn.commit()

        resp = client.get("/api/v1/runs", params={"page": 1, "page_size": 20})
        assert resp.status_code == 200
        payload = resp.json()
        items = payload["items"]
        assert items

        row = next((x for x in items if x["run_id"] == "r_null_1"), None)
        assert row is not None
        assert row["task_kind"] == ""
        assert row["experiment_id"] == ""
        assert row["asr_attack"] == 0.0
        assert row["risk_score"] == 0.0
        assert row["risk_level"] == "minimal"
        assert row["risk_scenario"] == "general"


# 验证 `运行记录 list merges cache and 产物 only 运行记录` 场景，防止相关行为在后续修改中退化。
def test_runs_list_merges_cache_and_artifact_only_runs(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        store = client.app.state.store

        cached_dir = tmp_path / "artifacts" / "runs" / "r_cached"
        cached_dir.mkdir(parents=True)
        (cached_dir / "summary.json").write_text('{"run_id":"r_cached","attack":"fgsm"}', encoding="utf-8")
        with store._connect() as conn:  # noqa: SLF001 - test-only direct DB seed
            conn.execute(
                """
                INSERT INTO runs_cache(
                    run_id, created_at, task_kind, dataset_name, benchmark_tag, attack, mode, defense, experiment_id,
                    model_adapter, asr, asr_attack, asr_defended, defense_gain, avg_l2, path
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "r_cached",
                    "2026-05-02T01:00:00+00:00",
                    "vlr",
                    "coco_subset",
                    "cached",
                    "fgsm",
                    "A",
                    "none",
                    "cached_job",
                    "clip_hf",
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    1.0,
                    str(cached_dir),
                ),
            )
            conn.commit()

        artifact_dir = tmp_path / "artifacts" / "runs" / "r_artifact_only"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "summary.json").write_text(
            '{"run_id":"r_artifact_only","created_at":"2026-05-02T02:00:00+00:00",'
            '"task_kind":"vlr","dataset_name":"coco_subset","attack":"advedm_plus",'
            '"asr":0.2,"asr_attack":0.2,"risk_level":"high","risk_score":0.6}',
            encoding="utf-8",
        )

        resp = client.get("/api/v1/runs", params={"page": 1, "page_size": 20})
        assert resp.status_code == 200
        run_ids = {row["run_id"] for row in resp.json()["items"]}

        assert "r_cached" in run_ids
        assert "r_artifact_only" in run_ids


# 验证 `运行记录 analytics groups 模型 风险 across all filtered 运行记录` 场景，防止相关行为在后续修改中退化。
def test_run_analytics_groups_model_risk_across_all_filtered_runs(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        for idx, asr in enumerate((0.2, 0.6, 0.8), start=1):
            run_dir = tmp_path / "artifacts" / "runs" / f"r_model_{idx}"
            run_dir.mkdir(parents=True)
            run_dir.joinpath("summary.json").write_text(
                (
                    '{"run_id":"r_model_%d","created_at":"2026-05-02T0%d:00:00+00:00",'
                    '"task_kind":"vlr","dataset_name":"coco_subset","attack":"advedm_plus",'
                    '"model_adapter":"clip_hf","victim_model_adapters":["clip_hf"],'
                    '"asr":%.1f,"asr_attack":%.1f,"avg_l2":0.0,"semantic_preservation_score":1.0}'
                )
                % (idx, idx, asr, asr),
                encoding="utf-8",
            )

        resp = client.get("/api/v1/runs/analytics", params={"exclude_demo": True})
        assert resp.status_code == 200
        group = next(item for item in resp.json()["model_risk_groups"] if item["model_adapter"] == "clip_hf")

        assert group["count"] == 3
        assert group["avg_risk_score"] == 0.28
        assert group["max_risk_score"] == 0.42
        assert group["medium_risk_count"] == 1
        assert group["low_risk_count"] == 2


# 验证 `运行记录 list hides advclip training 运行记录` 场景，防止相关行为在后续修改中退化。
def test_runs_list_hides_advclip_training_runs(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        train_dir = tmp_path / "artifacts" / "runs" / "r_train"
        train_dir.mkdir(parents=True)
        (train_dir / "summary.json").write_text(
            '{"run_id":"r_train","task_kind":"advclip_train","trained":true}',
            encoding="utf-8",
        )

        eval_dir = tmp_path / "artifacts" / "runs" / "r_eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "summary.json").write_text(
            '{"run_id":"r_eval","task_kind":"vlr","dataset_name":"coco_subset","attack":"pgd"}',
            encoding="utf-8",
        )

        resp = client.get("/api/v1/runs", params={"page": 1, "page_size": 20})
        assert resp.status_code == 200
        run_ids = {row["run_id"] for row in resp.json()["items"]}

        assert "r_eval" in run_ids
        assert "r_train" not in run_ids


# 验证 `generated only 运行记录 do not enter evaluation indexes` 场景，防止相关行为在后续修改中退化。
def test_generated_only_runs_do_not_enter_evaluation_indexes(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        generated_dir = tmp_path / "artifacts" / "runs" / "r_generated_only"
        generated_dir.mkdir(parents=True)
        generated_dir.joinpath("summary.json").write_text(
            """
            {
              "run_id": "r_generated_only",
              "created_at": "2026-05-07T07:36:13+00:00",
              "task_kind": "vlr",
              "dataset_name": "coco_subset",
              "attack": "fgsm",
              "surrogate_model_adapter": "clip_hf",
              "victim_model_adapters": [],
              "sample_generation_only": true,
              "requires_evaluation": true,
              "result_type": "generated_only",
              "risk_score": 0.0,
              "risk_level": ""
            }
            """,
            encoding="utf-8",
        )
        generated_dir.joinpath("report_data.json").write_text(
            '{"generation_only": true, "summary": {"run_id": "r_generated_only", "sample_generation_only": true}}',
            encoding="utf-8",
        )
        generated_dir.joinpath("cases_index.jsonl").write_text(
            '{"run_id":"r_generated_only","sample_id":"38","artifact_status":"generated_only","requires_evaluation":true}\n',
            encoding="utf-8",
        )

        eval_dir = tmp_path / "artifacts" / "runs" / "r_eval_ready"
        eval_dir.mkdir(parents=True)
        eval_dir.joinpath("summary.json").write_text(
            """
            {
              "run_id": "r_eval_ready",
              "created_at": "2026-05-07T08:00:00+00:00",
              "task_kind": "vlr",
              "dataset_name": "coco_subset",
              "attack": "fgsm",
              "model_adapter": "clip_hf",
              "victim_model_adapters": ["clip_hf"],
              "asr": 0.5,
              "asr_attack": 0.5
            }
            """,
            encoding="utf-8",
        )
        eval_dir.joinpath("cases_index.jsonl").write_text(
            '{"run_id":"r_eval_ready","sample_id":"38","artifact_status":"complete","judge_success":true}\n',
            encoding="utf-8",
        )

        runs_resp = client.get("/api/v1/runs", params={"page": 1, "page_size": 20, "exclude_demo": True})
        assert runs_resp.status_code == 200
        run_ids = {row["run_id"] for row in runs_resp.json()["items"]}
        assert "r_eval_ready" in run_ids
        assert "r_generated_only" not in run_ids

        analytics_resp = client.get("/api/v1/runs/analytics", params={"exclude_demo": True})
        assert analytics_resp.status_code == 200
        analytics = analytics_resp.json()
        assert analytics["total_runs"] == 1
        assert all(row.get("run_id") != "r_generated_only" for row in analytics["latest_runs"])

        cases_resp = client.get("/api/v1/runs/cases", params={"page": 1, "page_size": 20, "exclude_demo": True})
        assert cases_resp.status_code == 200
        case_run_ids = {row["run_id"] for row in cases_resp.json()["items"]}
        assert "r_eval_ready" in case_run_ids
        assert "r_generated_only" not in case_run_ids

        for suffix in ("summary", "results", "report-data", "cases", "cases/38"):
            detail_resp = client.get(f"/api/v1/runs/r_generated_only/{suffix}")
            assert detail_resp.status_code == 404


# 验证 `fake 模型 运行记录 do not enter any 运行记录 indexes` 场景，防止相关行为在后续修改中退化。
def test_fake_model_runs_do_not_enter_any_run_indexes(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        for run_id, model_adapter, sample_id in (
            ("r_fixture_fake", "fixture_vlm", "vqa-mini-0000"),
            ("r_dummy_fake", "dummy", "seed-000"),
        ):
            run_dir = tmp_path / "artifacts" / "runs" / run_id
            case_dir = run_dir / "cases" / sample_id
            case_dir.mkdir(parents=True)
            run_dir.joinpath("summary.json").write_text(
                (
                    '{"run_id":"%s","created_at":"2026-05-07T09:00:00+00:00",'
                    '"task_kind":"vqa","dataset_name":"generation_jsonl","benchmark_tag":"generation_vqa_smoke",'
                    '"attack":"advedm_plus","model_adapter":"%s","victim_model_adapters":["%s"],"asr":1.0}'
                )
                % (run_id, model_adapter, model_adapter),
                encoding="utf-8",
            )
            run_dir.joinpath("report_data.json").write_text(
                '{"summary":{"model_adapter":"%s","victim_model_adapters":["%s"]}}' % (model_adapter, model_adapter),
                encoding="utf-8",
            )
            run_dir.joinpath("cases_index.jsonl").write_text(
                '{"run_id":"%s","sample_id":"%s","artifact_status":"complete","judge_success":true}\n' % (run_id, sample_id),
                encoding="utf-8",
            )
            case_dir.joinpath("case_bundle.json").write_text(
                '{"task_kind":"vqa","model_tag":"%s","judge":{"success":true}}' % model_adapter,
                encoding="utf-8",
            )

        eval_dir = tmp_path / "artifacts" / "runs" / "r_real_eval"
        eval_dir.mkdir(parents=True)
        eval_dir.joinpath("summary.json").write_text(
            """
            {
              "run_id": "r_real_eval",
              "created_at": "2026-05-07T10:00:00+00:00",
              "task_kind": "vlr",
              "dataset_name": "coco_subset",
              "attack": "fgsm",
              "model_adapter": "clip_hf",
              "victim_model_adapters": ["clip_hf"],
              "asr": 0.0
            }
            """,
            encoding="utf-8",
        )
        eval_dir.joinpath("cases_index.jsonl").write_text(
            '{"run_id":"r_real_eval","sample_id":"38","artifact_status":"complete","judge_success":false}\n',
            encoding="utf-8",
        )

        runs_resp = client.get("/api/v1/runs", params={"page": 1, "page_size": 20})
        assert runs_resp.status_code == 200
        run_ids = {row["run_id"] for row in runs_resp.json()["items"]}
        assert "r_real_eval" in run_ids
        assert "r_fixture_fake" not in run_ids
        assert "r_dummy_fake" not in run_ids

        analytics_resp = client.get("/api/v1/runs/analytics")
        assert analytics_resp.status_code == 200
        assert all(row.get("run_id") not in {"r_fixture_fake", "r_dummy_fake"} for row in analytics_resp.json()["latest_runs"])

        cases_resp = client.get("/api/v1/runs/cases", params={"page": 1, "page_size": 20})
        assert cases_resp.status_code == 200
        case_run_ids = {row["run_id"] for row in cases_resp.json()["items"]}
        assert "r_real_eval" in case_run_ids
        assert "r_fixture_fake" not in case_run_ids
        assert "r_dummy_fake" not in case_run_ids

        for run_id, sample_id in (("r_fixture_fake", "vqa-mini-0000"), ("r_dummy_fake", "seed-000")):
            for suffix in ("summary", "results", "report-data", "cases", f"cases/{sample_id}", "assets/summary.json"):
                detail_resp = client.get(f"/api/v1/runs/{run_id}/{suffix}")
                assert detail_resp.status_code == 404


# 验证 `产物 运行记录 created at uses iso timestamp 来源 运行记录 id` 场景，防止相关行为在后续修改中退化。
def test_artifact_run_created_at_uses_iso_timestamp_from_run_id(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        run_id = "20260502_071613_559213"
        run_dir = tmp_path / "artifacts" / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            '{"run_id":"20260502_071613_559213","task_kind":"vlr","dataset_name":"coco_subset","attack":"advedm_plus"}',
            encoding="utf-8",
        )

        resp = client.get("/api/v1/runs", params={"page": 1, "page_size": 20})
        assert resp.status_code == 200
        row = next(item for item in resp.json()["items"] if item["run_id"] == run_id)

        assert row["created_at"].startswith("2026-05-02T07:16:13")


# 验证 `运行记录 list exposes 图文检索 dashboard 指标` 场景，防止相关行为在后续修改中退化。
def test_runs_list_exposes_vlr_dashboard_metrics(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        run_id = "r_vlr_metrics"
        run_dir = tmp_path / "artifacts" / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            """
            {
              "run_id": "r_vlr_metrics",
              "task_kind": "vlr",
              "dataset_name": "coco_subset",
              "attack": "advedm_plus",
              "num_images": 16,
              "num_texts": 16,
              "surrogate_model_adapter": "clip_hf",
              "victim_model_adapters": ["clip_hf"],
              "victims": {
                "clip_hf": {
                  "clean": {"ir_r@1": 0.875, "tr_r@1": 0.8125, "mean_rank_ir": 1.125, "mean_rank_tr": 1.25},
                  "attacked": {"ir_r@1": 0.25, "tr_r@1": 0.25, "mean_rank_ir": 3.1875, "mean_rank_tr": 3.3125},
                  "conditional": {"ir_rank_delta_mean": 2.0625, "tr_rank_delta_mean": 2.0625}
                }
              }
            }
            """,
            encoding="utf-8",
        )

        resp = client.get("/api/v1/runs", params={"page": 1, "page_size": 20})
        assert resp.status_code == 200
        row = next(item for item in resp.json()["items"] if item["run_id"] == run_id)

        assert row["sample_pair_count"] == 256
        assert row["surrogate_model_adapter"] == "clip_hf"
        assert row["victim_model_adapters"] == ["clip_hf"]
        assert row["clean_r1_mean"] == 0.84375
        assert row["attacked_r1_mean"] == 0.25
        assert row["attack_drop_r1_mean"] == 0.59375
        assert row["rank_delta_mean"] == 2.0625


# 验证 `运行记录 list exposes 生成式评测 dashboard 指标` 场景，防止相关行为在后续修改中退化。
def test_runs_list_exposes_generation_dashboard_metrics(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        run_id = "r_generation_metrics"
        run_dir = tmp_path / "artifacts" / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            """
            {
              "run_id": "r_generation_metrics",
              "task_kind": "vqa",
              "dataset_name": "generation_jsonl",
              "benchmark_tag": "coco_object_probe_val_real",
              "attack": "advedm_plus",
              "model_adapter": "openai_qwen35_9b",
              "victim_model_adapters": ["openai_qwen35_9b"],
              "generation_metrics": {
                "clean_accuracy": 1.0,
                "attacked_accuracy": 0.5,
                "answer_change_rate": 0.25,
                "target_flip_rate": 0.5,
                "semantic_preservation_rate": 0.8
              }
            }
            """,
            encoding="utf-8",
        )

        resp = client.get("/api/v1/runs", params={"page": 1, "page_size": 20})
        assert resp.status_code == 200
        row = next(item for item in resp.json()["items"] if item["run_id"] == run_id)

        assert row["clean_accuracy"] == 1.0
        assert row["attacked_accuracy"] == 0.5
        assert row["answer_change_rate"] == 0.25
        assert row["target_flip_rate"] == 0.5
        assert row["semantic_preservation_rate"] == 0.8
