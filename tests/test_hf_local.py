import pytest

from mmsec_eval.model_adapters.hf_local import hf_load_failure_message, require_cuda_device


def test_hf_load_failure_message_explains_local_only_recovery() -> None:
    message = hf_load_failure_message(
        adapter_label="CLIP",
        model_name="openai/clip-vit-base-patch32",
        source="artifacts/hf_models/clip",
        device="cuda",
        local_only=True,
        cause=FileNotFoundError("missing"),
    )

    assert "Failed to load CLIP model" in message
    assert "source=artifacts/hf_models/clip" in message
    assert "MMSEC_HF_LOCAL_ONLY=0" in message
    assert "Root cause: missing" in message


def test_hf_load_failure_message_explains_online_recovery() -> None:
    message = hf_load_failure_message(
        adapter_label="ViLT",
        model_name="dandelin/vilt-b32-finetuned-coco",
        source="dandelin/vilt-b32-finetuned-coco",
        device="cuda",
        local_only=False,
        cause=RuntimeError("network"),
    )

    assert "Check model id/network/HF auth" in message
    assert "MMSEC_HF_LOCAL_ONLY=1" in message


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, cuda_version: str | None, available: bool) -> None:
        self.version = type("Version", (), {"cuda": cuda_version})()
        self.cuda = _FakeCuda(available)


def test_require_cuda_device_accepts_available_cuda(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cuda")

    assert require_cuda_device("Adapter", _FakeTorch("12.1", True)) == "cuda"


def test_require_cuda_device_rejects_cpu_runtime(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cpu")

    with pytest.raises(RuntimeError, match="requires CUDA runtime device"):
        require_cuda_device("Adapter", _FakeTorch("12.1", True))


def test_require_cuda_device_rejects_torch_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cuda")

    with pytest.raises(RuntimeError, match="CUDA-enabled torch"):
        require_cuda_device("Adapter", _FakeTorch(None, False))
