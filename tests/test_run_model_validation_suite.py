from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path("scripts/run_model_validation_suite.py").resolve()
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_model_validation_suite", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_existing_rows_dedupes_by_identity_and_keeps_supplementary_runs(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "model_adapter": "clip_hf",
                        "attack": "fgsm",
                        "experiment_id": "scientific_validation_clip_hf_fgsm",
                        "job_status": "failed",
                        "run_id": "",
                    },
                    {
                        "model_adapter": "clip_hf",
                        "attack": "fgsm",
                        "experiment_id": "scientific_validation_clip_hf_fgsm",
                        "job_status": "success",
                        "run_id": "r1",
                    },
                    {
                        "model_adapter": "clip_hf",
                        "attack": "fgsm",
                        "experiment_id": "scientific_validation_clip_hf_fgsm_single",
                        "job_status": "success",
                        "run_id": "r1s",
                    },
                    {"model_adapter": "openai_qwen3_vl", "attack": "advedm_plus", "job_status": "success", "run_id": "r2"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = module._load_existing_rows(path)

    assert len(rows) == 3
    assert rows[0]["job_status"] == "success"
    assert rows[0]["run_id"] == "r1"
    assert rows[1]["experiment_id"] == "scientific_validation_clip_hf_fgsm_single"


def test_successful_keys_only_marks_canonical_success_rows():
    module = _load_module()
    rows = [
        {
            "model_adapter": "clip_hf",
            "attack": "fgsm",
            "experiment_id": "scientific_validation_clip_hf_fgsm",
            "job_status": "success",
        },
        {
            "model_adapter": "clip_hf",
            "attack": "advedm_plus",
            "experiment_id": "scientific_validation_clip_hf_advedm_plus_single",
            "job_status": "success",
        },
        {"model_adapter": "clip_hf", "attack": "advedm_plus", "job_status": "failed"},
        {"model_adapter": "openai_qwen3_vl", "attack": "fgsm", "job_status": "queued"},
    ]

    keys = module._successful_keys(rows)

    assert keys == {("clip_hf", "fgsm")}


def test_upsert_row_replaces_failed_result_for_same_key():
    module = _load_module()
    rows = [
        {
            "model_adapter": "clip_hf",
            "attack": "fgsm",
            "experiment_id": "scientific_validation_clip_hf_fgsm",
            "job_status": "failed",
            "run_id": "",
        },
        {"model_adapter": "openai_qwen3_vl", "attack": "fgsm", "job_status": "success", "run_id": "r2"},
    ]

    module._upsert_row(
        rows,
        {
            "model_adapter": "clip_hf",
            "attack": "fgsm",
            "experiment_id": "scientific_validation_clip_hf_fgsm",
            "job_status": "success",
            "run_id": "r1",
        },
    )

    assert len(rows) == 2
    assert rows[0]["job_status"] == "success"
    assert rows[0]["run_id"] == "r1"


def test_payload_sets_runtime_device():
    module = _load_module()

    payload = module._payload(
        dataset_name="flickr1k",
        attack="fgsm",
        model_adapter="clip_hf",
        experiment_id="scientific_validation_clip_hf_fgsm",
        seed=20260418,
        runtime_device="cuda:3",
        max_pairs=256,
        openai_timeout_seconds=180,
    )

    assert payload["override"]["runtime"]["device"] == "cuda:3"
    assert payload["override"]["runner"]["max_pairs"] == 256
    assert payload["override"]["model"]["openai_timeout"] == 180


def test_extract_run_evidence_captures_clean_attack_defense_metrics():
    module = _load_module()

    evidence = module._extract_run_evidence(
        {
            "victim_compare": [
                {
                    "clean": {"ir_r@1": 0.8, "tr_r@1": 0.6},
                    "attacked": {"ir_r@1": 0.3, "tr_r@1": 0.5},
                    "delta_mean_rank_ir": 1.0,
                    "delta_mean_rank_tr": 0.5,
                }
            ],
            "defense_compare": [
                {
                    "defense_recovery_r1": 0.2,
                    "defense_utility_drop@1": 0.05,
                }
            ],
        }
    )

    assert evidence["clean_r1_mean"] == pytest.approx(0.7)
    assert evidence["attacked_r1_mean"] == pytest.approx(0.4)
    assert evidence["attack_drop_r1_mean"] == pytest.approx(0.3)
    assert evidence["defense_recovery_r1_mean"] == pytest.approx(0.2)
    assert evidence["defense_utility_drop_r1_mean"] == pytest.approx(0.05)
    assert evidence["mean_rank_delta_mean"] == pytest.approx(0.75)


def test_summarize_requires_nontrivial_attack_signal():
    module = _load_module()
    rows = [
        {
            "model_adapter": "clip_hf",
            "attack": "fgsm",
            "dataset_name": "flickr1k",
            "experiment_id": "scientific_validation_clip_hf_fgsm",
            "job_status": "success",
            "num_victim_failures": 0,
            "asr_attack": 0.0,
            "asr_defended": 0.0,
            "defense_gain": 0.0,
            "clean_r1_mean": 0.8,
            "attack_drop_r1_mean": 0.0,
            "defense_recovery_r1_mean": 0.0,
        },
        {
            "model_adapter": "clip_hf",
            "attack": "advedm_plus",
            "dataset_name": "flickr1k",
            "experiment_id": "scientific_validation_clip_hf_advedm_plus",
            "job_status": "success",
            "num_victim_failures": 0,
            "asr_attack": 0.11,
            "asr_defended": 0.04,
            "defense_gain": 0.07,
            "clean_r1_mean": 0.8,
            "attack_drop_r1_mean": 0.12,
            "defense_recovery_r1_mean": 0.08,
        },
    ]

    summary = module._summarize(
        rows,
        attacks=["fgsm", "advedm_plus"],
        dataset_name="flickr1k",
        required_model_count=1,
        max_pairs=256,
        minimum_qualifying_attack_count_per_model=1,
    )

    assert summary["passed"] is True
    assert summary["scientific_quality_passed"] is True
    assert summary["validated_models"] == ["clip_hf"]
    assert summary["per_model"][0]["scientific_signal_ok"] is True
    assert summary["per_model"][0]["clean_baseline_ok"] is True
    assert summary["per_model"][0]["retrieval_drop_signal_ok"] is True
    assert summary["per_model"][0]["scientific_quality_ok"] is True
    assert summary["per_model"][0]["qualifying_attacks"] == ["advedm_plus"]
    assert summary["benchmark_attack_coverage_ok"] is True
    assert summary["criterion"]["minimum_attack_asr_any"] == module.MIN_ATTACK_ASR_ANY
    assert summary["criterion"]["minimum_qualifying_attack_count_per_model"] == 1
    assert summary["criterion"]["minimum_clean_r1_mean"] == module.MIN_CLEAN_R1_MEAN


def test_summarize_rejects_model_without_any_qualifying_attack():
    module = _load_module()
    rows = [
        {
            "model_adapter": "clip_hf",
            "attack": "fgsm",
            "dataset_name": "flickr1k",
            "experiment_id": "scientific_validation_clip_hf_fgsm",
            "job_status": "success",
            "num_victim_failures": 0,
            "asr_attack": 0.0,
            "attack_drop_r1_mean": 0.0,
        },
        {
            "model_adapter": "clip_hf",
            "attack": "advedm_plus",
            "dataset_name": "flickr1k",
            "experiment_id": "scientific_validation_clip_hf_advedm_plus",
            "job_status": "success",
            "num_victim_failures": 0,
            "asr_attack": 0.0,
            "attack_drop_r1_mean": 0.0,
        },
    ]

    summary = module._summarize(
        rows,
        attacks=["fgsm", "advedm_plus"],
        dataset_name="flickr1k",
        required_model_count=1,
        max_pairs=256,
        minimum_qualifying_attack_count_per_model=1,
    )

    assert summary["passed"] is False
    assert summary["validated_models"] == []
    assert summary["per_model"][0]["qualifying_attack_count"] == 0


def test_summarize_can_scope_to_selected_models():
    module = _load_module()
    summary = module._summarize(
        [],
        attacks=["fgsm"],
        dataset_name="flickr1k",
        required_model_count=1,
        max_pairs=16,
        minimum_qualifying_attack_count_per_model=1,
        model_adapters=["openai_qwen3_vl"],
    )

    assert [row["model_adapter"] for row in summary["per_model"]] == ["openai_qwen3_vl"]
    assert summary["missing_models"] == ["openai_qwen3_vl"]


def test_summarize_ignores_supplementary_success_when_primary_run_failed():
    module = _load_module()
    rows = [
        {
            "model_adapter": "openai_gemma3_12b",
            "attack": "fgsm",
            "dataset_name": "flickr1k",
            "experiment_id": "scientific_validation_openai_gemma3_12b_fgsm",
            "job_status": "failed",
            "num_victim_failures": 0,
        },
        {
            "model_adapter": "openai_gemma3_12b",
            "attack": "fgsm",
            "dataset_name": "flickr1k",
            "experiment_id": "scientific_validation_openai_gemma3_12b_fgsm_single",
            "job_status": "success",
            "num_victim_failures": 0,
            "asr_attack": 0.75,
            "attack_drop_r1_mean": 0.0,
            "clean_r1_mean": 0.25,
        },
    ]

    summary = module._summarize(
        rows,
        attacks=["fgsm", "advedm_plus"],
        dataset_name="flickr1k",
        required_model_count=1,
        max_pairs=256,
        minimum_qualifying_attack_count_per_model=1,
    )

    assert summary["passed"] is False
    assert summary["validated_models"] == []
    assert summary["failed_row_count"] == 1
    assert summary["supplementary_row_count"] == 1
    assert summary["supplementary_rows"][0]["experiment_id"].endswith("_single")


def test_summarize_marks_low_clean_baseline_as_not_scientific_quality():
    module = _load_module()
    rows = [
        {
            "model_adapter": "openai_qwen35_9b",
            "attack": "fgsm",
            "dataset_name": "flickr1k",
            "experiment_id": "scientific_validation_openai_qwen35_9b_fgsm",
            "job_status": "success",
            "num_victim_failures": 0,
            "asr_attack": 0.9375,
            "attack_drop_r1_mean": 0.0,
            "clean_r1_mean": 0.0625,
        },
        {
            "model_adapter": "openai_qwen35_9b",
            "attack": "advedm_plus",
            "dataset_name": "flickr1k",
            "experiment_id": "scientific_validation_openai_qwen35_9b_advedm_plus",
            "job_status": "failed",
            "num_victim_failures": 0,
        },
    ]

    summary = module._summarize(
        rows,
        attacks=["fgsm", "advedm_plus"],
        dataset_name="flickr1k",
        required_model_count=1,
        max_pairs=256,
        minimum_qualifying_attack_count_per_model=1,
    )

    assert summary["validated_models"] == ["openai_qwen35_9b"]
    assert summary["scientific_quality_validated_models"] == []
    assert summary["per_model"][0]["clean_baseline_ok"] is False
    assert summary["per_model"][0]["scientific_quality_ok"] is False


def test_hydrate_row_from_run_summary_backfills_missing_evidence(tmp_path: Path, monkeypatch):
    module = _load_module()
    run_dir = tmp_path / "artifacts" / "runs" / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "victim_compare": [
                    {
                        "clean": {"ir_r@1": 0.9, "tr_r@1": 0.7},
                        "attacked": {"ir_r@1": 0.5, "tr_r@1": 0.3},
                        "delta_mean_rank_ir": 1.2,
                        "delta_mean_rank_tr": 0.8,
                    }
                ],
                "defense_compare": [
                    {
                        "defense_recovery_r1": 0.25,
                        "defense_utility_drop@1": 0.1,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    row = {"job_status": "success", "run_id": "run123"}

    hydrated = module._hydrate_row_from_run_summary(row)

    assert hydrated["attack_drop_r1_mean"] == pytest.approx(0.4)
    assert hydrated["defense_recovery_r1_mean"] == pytest.approx(0.25)


def test_local_vlm_launcher_skips_classic_adapter(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    event = module._launch_local_vlm("clip_hf", startup_timeout_seconds=1, poll_seconds=0.1)

    assert event["adapter"] == "clip_hf"
    assert event["launched"] is False
    assert event["reason"] == "classic_adapter"
    assert event["cleanup"]["stopped"] is True


def test_run_validation_jobs_records_local_vlm_launch_event(tmp_path: Path, monkeypatch):
    module = _load_module()
    calls: list[str] = []

    def fake_launch(model_adapter: str, *, startup_timeout_seconds: int, poll_seconds: float):
        calls.append(model_adapter)
        return {"adapter": model_adapter, "launched": True}

    def fake_run_one(args, api, row):
        row["job_status"] = "success"
        row["run_id"] = f"{row['model_adapter']}-{row['attack']}"
        return row

    monkeypatch.setattr(module, "_launch_local_vlm", fake_launch)
    monkeypatch.setattr(module, "_run_one_validation_job", fake_run_one)

    rows: list[dict] = []
    status = {}
    args = SimpleNamespace(
        no_auto_launch_local_vlm=False,
        local_vlm_startup_timeout_seconds=12,
        poll_seconds=0.1,
        dataset="flickr1k",
        seed_base=1,
    )
    module._run_validation_jobs(
        args,
        api=object(),
        attacks=["fgsm"],
        model_adapters=["clip_hf", "openai_qwen3_vl"],
        rows=rows,
        rows_path=tmp_path / "rows.json",
        status_path=tmp_path / "status.json",
        status=status,
    )

    assert calls == ["clip_hf", "openai_qwen3_vl"]
    assert [row["run_id"] for row in rows] == ["clip_hf-fgsm", "openai_qwen3_vl-fgsm"]
    assert [event["adapter"] for event in status["local_vlm_events"]] == ["clip_hf", "openai_qwen3_vl"]


def test_wait_for_validation_job_tolerates_transient_request_error(monkeypatch):
    module = _load_module()
    calls = {"count": 0}

    class FakeApi:
        def get_job(self, job_id: str):
            calls["count"] += 1
            if calls["count"] == 1:
                raise module.requests.ConnectionError("temporary disconnect")
            return {"id": job_id, "status": "success", "run_id": "run-ok"}

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    job = module._wait_for_validation_job(FakeApi(), "job-1", timeout_seconds=10, poll_seconds=0.1)

    assert calls["count"] == 2
    assert job["run_id"] == "run-ok"
