# 文件说明：该文件属于自动化测试，集中实现 test api jobs chain 相关逻辑。
from __future__ import annotations

from pathlib import Path

from mmsec_eval.cli import cmd_train_advclip

from api_test_utils import make_client, wait_job_done, write_toy_eval_cfg


# 中文注释：封装 _write_cfg 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _write_cfg(path: Path) -> None:
    write_toy_eval_cfg(path)


# 中文注释：封装 _write_cfg_vlr 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _write_cfg_vlr(path: Path) -> None:
    write_toy_eval_cfg(
        path,
        attack="advclip",
        num_samples=6,
        attack_config={
            "patch_size": 16,
            "patch_train_steps": 5,
            "steps": 1,
        },
        task={
            "kind": "vlr",
            "retrieval_k": [1, 5, 10],
            "eval_scope": "image",
        },
        runner={
            "max_samples": 6,
            "continue_on_error": False,
            "surrogate_model_adapter": "clip_hf",
            "victim_model_adapters": ["clip_hf"],
            "max_pairs": 0,
        },
    )


# 中文注释：封装 _write_cfg_train_advclip 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
def _write_cfg_train_advclip(path: Path) -> None:
    write_toy_eval_cfg(
        path,
        attack="advclip",
        attack_config={
            "mode": "A",
            "patch_size": 16,
            "use_gan": True,
            "gan_steps": 1,
            "patch_train_steps": 5,
            "steps": 1,
        },
        runner={
            "max_samples": 2,
            "continue_on_error": False,
            "surrogate_model_adapter": "clip_hf",
        },
    )


# 中文注释：验证 test_api_job_run_eval_chain 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_run_eval_chain(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cfg = tmp_path / "cfg.yaml"
        _write_cfg(cfg)

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_eval",
                "config_path": str(cfg),
                "override": {},
                "benchmark_mode": False,
            },
        )
        assert resp.status_code == 200
        job = resp.json()
        job_id = job["id"]

        status = wait_job_done(client, job_id, timeout_s=240.0)
        assert status == "success"
        done = client.get(f"/api/v1/jobs/{job_id}").json()
        run_id = done["run_id"]
        assert run_id

        s = client.get(f"/api/v1/runs/{run_id}/summary")
        assert s.status_code == 200
        r = client.get(f"/api/v1/runs/{run_id}/results")
        assert r.status_code == 200
        assert r.json()["total"] >= 1

        logs = client.get(f"/api/v1/jobs/{job_id}/logs")
        assert logs.status_code == 200
        assert logs.json()["total"] >= 1


# 中文注释：验证 test_api_job_run_vlr_chain 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_run_vlr_chain(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        # Train a patch first (synchronously): run-vlr must not use random init patches.
        train_cfg = tmp_path / "advclip_train_cfg.yaml"
        _write_cfg_train_advclip(train_cfg)
        assert cmd_train_advclip(str(train_cfg)) == 0

        cfg = tmp_path / "vlr_cfg.yaml"
        _write_cfg_vlr(cfg)

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_vlr",
                "config_path": str(cfg),
                "override": {},
                "benchmark_mode": False,
            },
        )
        assert resp.status_code == 200
        job = resp.json()
        job_id = job["id"]

        status = wait_job_done(client, job_id, timeout_s=240.0)
        assert status == "success"
        done = client.get(f"/api/v1/jobs/{job_id}").json()
        run_id = done["run_id"]
        assert run_id

        s = client.get(f"/api/v1/runs/{run_id}/summary")
        assert s.status_code == 200
        assert s.json().get("task_kind") == "vlr"

        logs = client.get(f"/api/v1/jobs/{job_id}/logs")
        assert logs.status_code == 200
        assert logs.json()["total"] >= 1


# 中文注释：验证 test_api_job_rejects_incompatible_surrogate_for_joint_attack 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_rejects_incompatible_surrogate_for_joint_attack(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cfg = tmp_path / "vlr_cfg.yaml"
        _write_cfg_vlr(cfg)

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_vlr",
                "config_path": str(cfg),
                "override": {
                    "plugins": {"attack": "advedm_plus", "model_adapter": "openai_qwen3_vl"},
                    "runner": {
                        "surrogate_model_adapter": "openai_qwen3_vl",
                        "victim_model_adapters": ["clip_hf", "openai_qwen3_vl"],
                    },
                },
                "benchmark_mode": False,
            },
        )

    assert resp.status_code == 422
    assert "当前只支持 clip_hf 作为代理模型" in resp.json()["detail"]


# 中文注释：验证 test_api_job_rejects_openai_surrogate_for_classic_gradient_attack 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_rejects_openai_surrogate_for_classic_gradient_attack(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cfg = tmp_path / "vlr_cfg.yaml"
        _write_cfg_vlr(cfg)

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_vlr",
                "config_path": str(cfg),
                "override": {
                    "plugins": {"attack": "fgsm", "model_adapter": "openai_qwen25_vl"},
                    "runner": {
                        "surrogate_model_adapter": "openai_qwen25_vl",
                        "victim_model_adapters": ["openai_qwen25_vl"],
                    },
                },
                "benchmark_mode": False,
            },
        )

        assert resp.status_code == 422
        assert "经典梯度攻击当前只支持具备 score_pairs_torch 的本地代理模型" in resp.json()["detail"]


# 中文注释：验证 test_api_job_rejects_fixture_vlr_victim_at_route 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_rejects_fixture_vlr_victim_at_route(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cfg = tmp_path / "vlr_fixture_cfg.yaml"
        _write_cfg_vlr(cfg)

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_vlr",
                "config_path": str(cfg),
                "override": {
                    "plugins": {"attack": "advclip", "model_adapter": "fixture_vlm"},
                    "runner": {
                        "surrogate_model_adapter": "clip_hf",
                        "victim_model_adapters": ["fixture_vlm"],
                    },
                },
                "benchmark_mode": False,
            },
        )

        assert resp.status_code == 422
        assert "fixture_vlm" in resp.json()["detail"]
        assert "不支持 VLR 图文检索真实测评" in resp.json()["detail"]


# 中文注释：验证 test_api_job_rejects_configured_fixture_generation_model_without_override 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_rejects_configured_fixture_generation_model_without_override(tmp_path: Path, monkeypatch):
    import yaml

    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cases = tmp_path / "vqa_v2_coco_val.jsonl"
        cases.write_text('{"id":"vqa-1","image":"x.jpg","question":"Is there a dog?","answer":"yes"}' + "\n", encoding="utf-8")
        cfg = tmp_path / "fixture_vqa_cfg.yaml"
        write_toy_eval_cfg(
            cfg,
            attack="advedm_plus",
            task={"kind": "vqa", "eval_scope": "image", "cases_jsonl": str(cases)},
            runner={"max_samples": 1, "surrogate_model_adapter": "clip_hf", "victim_model_adapters": ["fixture_vlm"]},
        )
        raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        raw["plugins"]["model_adapter"] = "fixture_vlm"
        raw["dataset"] = {"kind": "generation_jsonl", "benchmark_tag": "vqa_v2_coco_val_real"}
        cfg.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_vqa",
                "config_path": str(cfg),
                "override": {},
                "benchmark_mode": False,
            },
        )

        assert resp.status_code == 422
        assert "fixture_vlm" in resp.json()["detail"]
        assert "不支持 VQA 生成式真实测评" in resp.json()["detail"]


# 中文注释：验证 test_api_job_rejects_mismatched_generation_dataset_for_task 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_rejects_mismatched_generation_dataset_for_task(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cases = tmp_path / "coco_caption_object_val.jsonl"
        cases.write_text('{"id":"cap-1","image":"x.jpg","reference_captions":["a dog"],"target_object":"dog"}\n', encoding="utf-8")
        cfg = tmp_path / "vqa_cfg.yaml"
        write_toy_eval_cfg(
            cfg,
            attack="advedm_plus",
            task={"kind": "vqa", "eval_scope": "image", "cases_jsonl": str(cases)},
            runner={"max_samples": 1, "surrogate_model_adapter": "clip_hf", "victim_model_adapters": ["openai_qwen35_9b"]},
        )

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_vqa",
                "config_path": str(cfg),
                "override": {
                    "dataset": {"kind": "generation_jsonl", "benchmark_tag": "coco_caption_object_val_real"},
                    "plugins": {"attack": "advedm_plus", "model_adapter": "openai_qwen35_9b"},
                },
                "benchmark_mode": False,
            },
        )

        assert resp.status_code == 422
        assert "VQA task requires a VQA JSONL dataset" in resp.json()["detail"]

        vqa_cases = tmp_path / "vqa_v2_coco_val.jsonl"
        vqa_cases.write_text('{"id":"vqa-1","image":"x.jpg","question":"Is there a dog?","answer":"yes"}' + "\n", encoding="utf-8")
        caption_cfg = tmp_path / "caption_cfg.yaml"
        write_toy_eval_cfg(
            caption_cfg,
            attack="advedm_plus",
            task={"kind": "caption", "eval_scope": "image", "cases_jsonl": str(vqa_cases)},
            runner={"max_samples": 1, "surrogate_model_adapter": "clip_hf", "victim_model_adapters": ["openai_qwen35_9b"]},
        )

        caption_resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_caption",
                "config_path": str(caption_cfg),
                "override": {
                    "dataset": {"kind": "generation_jsonl", "benchmark_tag": "vqa_v2_coco_val_real"},
                    "plugins": {"attack": "advedm_plus", "model_adapter": "openai_qwen35_9b"},
                },
                "benchmark_mode": False,
            },
        )

        assert caption_resp.status_code == 422
        assert "Caption task requires the COCO caption object JSONL" in caption_resp.json()["detail"]


# 中文注释：验证 test_api_job_rejects_unknown_override_field 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_rejects_unknown_override_field(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cfg = tmp_path / "cfg.yaml"
        _write_cfg(cfg)

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "run_vlr",
                "config_path": str(cfg),
                "overrides": {"plugins": {"model_adapter": "clip_hf"}},
                "benchmark_mode": False,
            },
        )

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any(item.get("loc") == ["body", "overrides"] for item in detail)


# 中文注释：验证 test_api_job_train_advclip_chain 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_api_job_train_advclip_chain(tmp_path: Path, monkeypatch):
    with make_client(tmp_path, monkeypatch, skip_model_preflight=True) as client:
        cfg = tmp_path / "advclip_train_cfg.yaml"
        _write_cfg_train_advclip(cfg)

        resp = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "train_advclip",
                "config_path": str(cfg),
                "override": {},
                "benchmark_mode": False,
            },
        )
        assert resp.status_code == 200
        job = resp.json()
        job_id = job["id"]

        status = wait_job_done(client, job_id, timeout_s=240.0)
        assert status == "success"
        done = client.get(f"/api/v1/jobs/{job_id}").json()
        run_id = done["run_id"]
        assert run_id

        s = client.get(f"/api/v1/runs/{run_id}/summary")
        assert s.status_code == 200
        assert s.json().get("task_kind") == "advclip_train"

        registry = tmp_path / "artifacts" / "advclip_patch_registry.json"
        assert registry.exists()
