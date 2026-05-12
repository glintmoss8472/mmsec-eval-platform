# 文件说明：该文件属于自动化测试，集中实现 test probe advclip official assets 相关逻辑。
from __future__ import annotations

import json
from io import BytesIO

from scripts.probe_advclip_official_assets import probe_cstcloud_share


# 定义 `_FakeResponse` 的状态和行为边界，供自动化测试在固定职责内复用。
class _FakeResponse(BytesIO):
    status = 200

    # 实现 `_FakeResponse.__enter__` 的对象行为，维护该类在自动化测试中的调用契约。
    def __enter__(self) -> "_FakeResponse":
        return self

    # 实现 `_FakeResponse.__exit__` 的对象行为，维护该类在自动化测试中的调用契约。
    def __exit__(self, *args: object) -> None:
        return None


# 验证 `探测 cstcloud share marks expired` 场景，防止相关行为在后续修改中退化。
def test_probe_cstcloud_share_marks_expired(monkeypatch) -> None:
    body = json.dumps({"stat": "ERR_SHARE_EXPIRED", "errText": "分享已过期"}).encode()

    # 执行 `fake urlopen` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def fake_urlopen(request, timeout):  # noqa: ANN001
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = probe_cstcloud_share("JqKbqGfTRs", timeout=1)

    assert result["status"] == "expired"
    assert "raw_data.rar" in result["required_payload"]
    assert "blocker" in result


# 验证 `探测 cstcloud share marks available` 场景，防止相关行为在后续修改中退化。
def test_probe_cstcloud_share_marks_available(monkeypatch) -> None:
    body = json.dumps({"stat": "OK", "name": "raw_data.rar"}).encode()

    # 执行 `fake urlopen` 辅助逻辑，保持自动化测试中的输入处理和结果输出一致。
    def fake_urlopen(request, timeout):  # noqa: ANN001
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = probe_cstcloud_share("JqKbqGfTRs", timeout=1)

    assert result["status"] == "available"
    assert result["parsed_response"]["stat"] == "OK"
