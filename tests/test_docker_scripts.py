from __future__ import annotations

from pathlib import Path


def test_docker_validate_never_prefetches_missing_models_by_default() -> None:
    content = Path("scripts", "docker_validate.sh").read_text(encoding="utf-8")
    assert "check_portable_assets.py" in content
    assert "MMSEC_PREFETCH_IF_MISSING:-1" not in content
    assert "docker_prefetch_assets.sh" not in content
    assert 'export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in content
    assert 'export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"' in content


def test_docker_build_and_publish_require_portable_asset_contract() -> None:
    build_content = Path("scripts", "docker_build_image.sh").read_text(encoding="utf-8")
    publish_content = Path("scripts", "docker_publish_artifact_registry.sh").read_text(encoding="utf-8")
    dockerfile_content = Path("Dockerfile").read_text(encoding="utf-8")

    assert 'check_portable_assets.py" --artifacts-root "${PROJECT_ROOT}/artifacts"' in build_content
    assert 'VALIDATE_BEFORE_PUSH="${VALIDATE_BEFORE_PUSH:-1}"' in publish_content
    assert "docker_run_offline_validation.sh" in publish_content
    assert "MMSEC_LOCAL_VLM_REQUIRE_OFFLINE=1" in dockerfile_content
    assert "portable_asset_manifest.json" in dockerfile_content


def test_portable_dockerfile_does_not_bundle_docs_payload() -> None:
    dockerfile_content = Path("Dockerfile").read_text(encoding="utf-8")

    assert "COPY docs ./docs" not in dockerfile_content
    assert "COPY seed ./seed" not in dockerfile_content
    assert "COPY artifacts/docs_index.json ./bundled/artifacts/docs_index.json" not in dockerfile_content
    assert "COPY artifacts/docs_snippets.jsonl ./bundled/artifacts/docs_snippets.jsonl" not in dockerfile_content
    assert "COPY seed/data ./seed/data" in dockerfile_content
    assert "COPY seed/runs ./seed/runs" in dockerfile_content


def test_local_vlm_runtime_requires_complete_offline_assets_when_requested() -> None:
    content = Path("scripts", "_local_vlm_server_env.sh").read_text(encoding="utf-8")

    assert "mmsec_local_model_ready" in content
    assert 'mmsec_truthy "${MMSEC_LOCAL_VLM_REQUIRE_OFFLINE:-0}"' in content
    assert "offline local VLM assets are incomplete" in content
