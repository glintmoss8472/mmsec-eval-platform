# 文件说明：该文件属于自动化测试，集中实现 test hf local 相关逻辑。
import pytest

from mmsec_eval.model_adapters.hf_local import hf_load_failure_message, require_cuda_device


# 中文注释：验证 test_hf_load_failure_message_explains_local_only_recovery 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
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


# 中文注释：验证 test_hf_load_failure_message_explains_online_recovery 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
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


# 中文注释：定义 _FakeCuda 的结构化职责，作为自动化测试中状态、配置或行为的边界。
class _FakeCuda:
    # 中文注释：封装 _FakeCuda.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self, available: bool) -> None:
        self._available = available

    # 中文注释：实现 _FakeCuda.is_available 的核心行为，维护自动化测试在该对象上的调用契约。
    def is_available(self) -> bool:
        return self._available


# 中文注释：定义 _FakeTorch 的结构化职责，作为自动化测试中状态、配置或行为的边界。
class _FakeTorch:
    # 中文注释：封装 _FakeTorch.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self, cuda_version: str | None, available: bool) -> None:
        self.version = type("Version", (), {"cuda": cuda_version})()
        self.cuda = _FakeCuda(available)


# 中文注释：验证 test_require_cuda_device_accepts_available_cuda 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_require_cuda_device_accepts_available_cuda(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cuda")

    assert require_cuda_device("Adapter", _FakeTorch("12.1", True)) == "cuda"


# 中文注释：验证 test_require_cuda_device_rejects_cpu_runtime 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_require_cuda_device_rejects_cpu_runtime(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cpu")

    with pytest.raises(RuntimeError, match="requires CUDA runtime device"):
        require_cuda_device("Adapter", _FakeTorch("12.1", True))


# 中文注释：验证 test_require_cuda_device_rejects_torch_without_cuda 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_require_cuda_device_rejects_torch_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("MMSEC_RUNTIME_DEVICE", "cuda")

    with pytest.raises(RuntimeError, match="CUDA-enabled torch"):
        require_cuda_device("Adapter", _FakeTorch(None, False))
