# 文件说明：该文件属于自动化测试，集中实现 test sample assets pending 相关逻辑。
from __future__ import annotations

from pathlib import Path

from api_test_utils import make_client


# 执行 `asset` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
def _asset(run_id: str, *, status: str, artifact_status: str, created_at: str) -> dict[str, object]:
    return {
        "asset_id": f"{run_id}::38",
        "variant_id": f"{run_id}::38::fgsm",
        "source_run_id": run_id,
        "source_case_id": "38",
        "task_kind": "vlr",
        "dataset_name": "coco_subset",
        "benchmark_tag": "coco_subset",
        "model_adapter": "" if status == "pending_evaluation" else "clip_hf",
        "attack": "fgsm",
        "attack_scope": "图像扰动",
        "source_text": "A black Honda motorcycle parked in front of a garage.",
        "target_text": "",
        "clean_image_ref": f"artifacts/runs/{run_id}/cases/38/clean.png",
        "adv_image_ref": f"artifacts/runs/{run_id}/cases/38/adv.png",
        "artifact_status": artifact_status,
        "reusable_status": status,
        "reusable_note": "pending" if status == "pending_evaluation" else "ready",
        "judge_success": status != "pending_evaluation",
        "risk_level": "" if status == "pending_evaluation" else "low",
        "risk_score": 0.0,
        "perturbation_l2": 5.9,
        "perturbation_linf": 0.008,
        "semantic_score": 0.0,
        "created_at": created_at,
        "metadata": {"sample_generation_only": status == "pending_evaluation"},
    }


# 验证 `pending 样本 batches are visible but not 报告 linked` 场景，防止相关行为在后续修改中退化。
def test_pending_sample_batches_are_visible_but_not_report_linked(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        store = client.app.state.store
        store.upsert_sample_assets(
            [
                _asset(
                    "20260507_153610_133285",
                    status="pending_evaluation",
                    artifact_status="generated_only",
                    created_at="2026-05-07T15:36:13+08:00",
                ),
                _asset(
                    "20260507_014757_693976",
                    status="ready",
                    artifact_status="complete",
                    created_at="2026-05-07T01:48:09+08:00",
                ),
            ]
        )

        default_resp = client.get("/api/v1/samples/batches", params={"page": 1, "page_size": 10})
        assert default_resp.status_code == 200
        payload = default_resp.json()
        assert payload["total"] == 2
        assert payload["summary"]["pending_evaluation_assets"] == 1
        assert payload["summary"]["callable_assets"] == 1
        pending = payload["items"][0]
        assert pending["batch_id"] == "20260507_153610_133285"
        assert pending["batch_status"] == "pending_evaluation"
        assert pending["report_url"] == ""
        assert pending["asset_ids"] == ["20260507_153610_133285::38"]
        assert pending["preview_assets"]
        for asset in pending["preview_assets"]:
            assert asset["case_url"] == ""
            assert asset["report_url"] == ""

        ready_resp = client.get("/api/v1/samples/batches", params={"reusable_status": "ready"})
        assert ready_resp.status_code == 200
        assert ready_resp.json()["total"] == 1
        assert ready_resp.json()["items"][0]["batch_id"] == "20260507_014757_693976"

        pending_resp = client.get("/api/v1/samples/batches", params={"reusable_status": "pending_evaluation"})
        assert pending_resp.status_code == 200
        assert pending_resp.json()["total"] == 1
        assert pending_resp.json()["items"][0]["batch_id"] == "20260507_153610_133285"

        pending_assets_resp = client.get("/api/v1/samples", params={"reusable_status": "pending_evaluation"})
        assert pending_assets_resp.status_code == 200
        pending_assets = pending_assets_resp.json()["items"]
        assert len(pending_assets) == 1
        assert pending_assets[0]["asset_id"] == "20260507_153610_133285::38"
        assert pending_assets[0]["report_url"] == ""
        assert pending_assets[0]["case_url"] == ""


# 验证 `fake 模型 样本 资产 are not imported or listed` 场景，防止相关行为在后续修改中退化。
def test_fake_model_sample_assets_are_not_imported_or_listed(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        store = client.app.state.store
        fake_asset = _asset(
            "20260503_221636_400614",
            status="ready",
            artifact_status="complete",
            created_at="2026-05-03T22:16:36+08:00",
        )
        fake_asset["model_adapter"] = "fixture_vlm"
        ready_asset = _asset(
            "20260507_014757_693976",
            status="ready",
            artifact_status="complete",
            created_at="2026-05-07T01:48:09+08:00",
        )

        written = store.upsert_sample_assets([fake_asset, ready_asset])

        assert written == 1
        assert store.count_sample_assets() == 1
        batches = client.get("/api/v1/samples/batches", params={"page": 1, "page_size": 10})
        assert batches.status_code == 200
        text = batches.text
        assert "20260503_221636_400614" not in text
        assert "fixture_vlm" not in text
        assert batches.json()["total"] == 1
        assert batches.json()["items"][0]["batch_id"] == "20260507_014757_693976"
