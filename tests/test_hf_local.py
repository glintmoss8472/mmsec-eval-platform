# 文件说明：该文件属于自动化测试，集中实现 test hf local 相关逻辑。
import pytest

from mmsec_eval.model_adapters.hf_local import hf_load_failure_message, require_cuda_device


# 验证 `Hugging Face load failure message explains 本地 only recovery` 场景，防止相关行为在后续修改中退化。
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


# 验证 `Hugging Face load failure message explains online recovery` 场景，防止相关行为在后续修改中退化。
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


# 实现 `_FakeCuda.__init__` 的对象行为，维护该类在自动化测试中的调用契约。
class _FakeCuda:
    # 封装 _FakeCuda.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self, available: bool) -> None:
        self._available = available

    # 判断 `是否 available` 条件是否成立，为调用方提供布尔决策。
    def is_available(self) -> bool:
        return self._available


# 实现 `_FakeTorch.__init__` 的对象行为，维护该类在自动化测试中的调用契约。
class _FakeTorch:
    # 封装 _FakeTorch.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self, cuda_version: str | None, available: bool) -> None:
        self.version = type("Version", (), {"cuda": cuda_version})()
        self.cuda = _FakeCuda(available)


# 验证 `require CUDA device accepts available CUDA` 场景，防止相关行为在后续修改中退化。
def test_require_cuda_device_accepts_available_cuda(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cuda")

    assert require_cuda_device("Adapter", _FakeTorch("12.1", True)) == "cuda"


# 验证 `require CUDA device rejects cpu runtime` 场景，防止相关行为在后续修改中退化。
def test_require_cuda_device_rejects_cpu_runtime(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cpu")

    with pytest.raises(RuntimeError, match="requires CUDA runtime device"):
        require_cuda_device("Adapter", _FakeTorch("12.1", True))


# 验证 `require CUDA device rejects PyTorch 移除 CUDA` 场景，防止相关行为在后续修改中退化。
def test_require_cuda_device_rejects_torch_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cuda")

    with pytest.raises(RuntimeError, match="CUDA-enabled torch"):
        require_cuda_device("Adapter", _FakeTorch(None, False))
