# 文件说明：该文件属于自动化测试，集中实现 test job executor dataset prepare 相关逻辑。
from __future__ import annotations

from types import SimpleNamespace


# 实现 `_DummyStore.__init__` 的对象行为，维护该类在自动化测试中的调用契约。
class _DummyStore:
    # 封装 _DummyStore.__init__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self.datasets: list[dict[str, object]] = []

    # 推断 `upsert 数据集`，从样本、配置或运行记录中提取统一名称。
    def upsert_dataset(self, **kwargs):
        self.datasets.append(kwargs)


# 验证 `Flickr1k 数据集 prepare uses unified Flickr30k slice defaults` 场景，防止相关行为在后续修改中退化。
def test_flickr1k_dataset_prepare_uses_unified_flickr30k_slice_defaults(monkeypatch):
    from mmsec_api.services.job_executor import JobExecutor

    store = _DummyStore()
    executor = JobExecutor(store=store)
    captured: dict[str, object] = {}

    # 执行 `fake 运行记录` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def fake_run(cmd, capture_output, text, cwd):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mmsec_api.services.job_executor.subprocess.run", fake_run)
    monkeypatch.setattr("mmsec_api.services.job_executor.platform.system", lambda: "Linux")

    result = executor._run_dataset_prepare({"name": "flickr1k"}, lambda *_args: None)
    cmd = captured["cmd"]

    assert "-Root" in cmd
    assert cmd[cmd.index("-Root") + 1] == "data/flickr30k"
    assert "-OutputFile" in cmd
    assert cmd[cmd.index("-OutputFile") + 1] == "captions_index_single.jsonl"
    assert "-MaxItems" in cmd
    assert cmd[cmd.index("-MaxItems") + 1] == "256"
    assert result["root_path"] == "data/flickr30k"
    assert store.datasets[-1]["root_path"] == "data/flickr30k"


# 验证 `mini Flickr 数据集 prepare registers demo fixture` 场景，防止相关行为在后续修改中退化。
def test_mini_flickr_dataset_prepare_registers_demo_fixture(tmp_path):
    from mmsec_api.services.job_executor import JobExecutor

    root = tmp_path / "mini_flickr"
    (root / "images").mkdir(parents=True)
    (root / "captions_index.jsonl").write_text('{"image":"0.jpg","caption":"a"}\n{"image":"1.jpg","caption":"b"}\n', encoding="utf-8")

    store = _DummyStore()
    executor = JobExecutor(store=store)

    result = executor._run_dataset_prepare({"name": "mini_flickr", "root_path": str(root)}, lambda *_args: None)

    assert result["dataset"] == "mini_flickr"
    assert result["prepared"] is True
    assert result["root_path"] == str(root)
    assert result["item_count"] == 2
    assert store.datasets[-1]["name"] == "mini_flickr"
    assert store.datasets[-1]["root_path"] == str(root)
    assert store.datasets[-1]["note"] == "prepared via api (demo fixture)"
