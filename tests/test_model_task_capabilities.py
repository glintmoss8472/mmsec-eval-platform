from __future__ import annotations

from mmsec_api.services.model_runtime import model_supports_task, task_capabilities_for_adapter


def test_fixture_is_not_formal_task_model() -> None:
    assert task_capabilities_for_adapter("fixture_vlm") == []
    assert model_supports_task("fixture_vlm", "vqa") is False
    assert model_supports_task("fixture_vlm", "caption") is False
    assert model_supports_task("fixture_vlm", "vlr") is False


def test_task_specific_model_capabilities() -> None:
    assert task_capabilities_for_adapter("clip_hf") == ["vlr"]
    assert model_supports_task("clip_hf", "vlr") is True
    assert model_supports_task("clip_hf", "vqa") is False
    assert model_supports_task("openai_qwen35_9b", "vlr") is True
    assert model_supports_task("openai_qwen35_9b", "vqa") is True
    assert model_supports_task("openai_qwen35_9b", "caption") is True
