# 文件说明：该文件属于自动化测试，集中实现 test probe advclip official assets 相关逻辑。
from __future__ import annotations

import json
from io import BytesIO

from scripts.probe_advclip_official_assets import probe_cstcloud_share


# 中文注释：定义 _FakeResponse 的结构化职责，作为自动化测试中状态、配置或行为的边界。
class _FakeResponse(BytesIO):
    status = 200

    # 中文注释：封装 _FakeResponse.__enter__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __enter__(self) -> "_FakeResponse":
        return self

    # 中文注释：封装 _FakeResponse.__exit__ 的内部步骤，让自动化测试主流程保持清晰并隔离边界细节。
    def __exit__(self, *args: object) -> None:
        return None


# 中文注释：验证 test_probe_cstcloud_share_marks_expired 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_probe_cstcloud_share_marks_expired(monkeypatch) -> None:
    body = json.dumps({"stat": "ERR_SHARE_EXPIRED", "errText": "分享已过期"}).encode()

    # 中文注释：实现 fake_urlopen 的核心流程，支撑自动化测试中的业务语义和异常边界。
    def fake_urlopen(request, timeout):  # noqa: ANN001
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = probe_cstcloud_share("JqKbqGfTRs", timeout=1)

    assert result["status"] == "expired"
    assert "raw_data.rar" in result["required_payload"]
    assert "blocker" in result


# 中文注释：验证 test_probe_cstcloud_share_marks_available 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_probe_cstcloud_share_marks_available(monkeypatch) -> None:
    body = json.dumps({"stat": "OK", "name": "raw_data.rar"}).encode()

    # 中文注释：实现 fake_urlopen 的核心流程，支撑自动化测试中的业务语义和异常边界。
    def fake_urlopen(request, timeout):  # noqa: ANN001
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = probe_cstcloud_share("JqKbqGfTRs", timeout=1)

    assert result["status"] == "available"
    assert result["parsed_response"]["stat"] == "OK"
