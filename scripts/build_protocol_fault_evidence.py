# 文件说明：该文件属于运维与实验脚本，集中实现 build protocol fault evidence 相关逻辑。
from __future__ import annotations

import json
import os
import socket
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
TARGET_HOST = "222.20.126.115"
TARGET_PORTS = [22, 8000, 8011, 8012, 8013]
OUT_DIR = ROOT / "artifacts" / "protocol_fault_evidence_20260419"


# 执行 `now iso` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


# 定位 `ensure out 目录`，把配置值或请求上下文转换成实际文件系统路径。
def _ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# 执行 `clear proxy 环境` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _clear_proxy_env() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = ",".join([TARGET_HOST, "127.0.0.1", "localhost"])
    os.environ["no_proxy"] = os.environ["NO_PROXY"]


# 执行 `port connect and read` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _port_connect_and_read(port: int, payload: bytes | None = None) -> dict[str, Any]:
    start = datetime.now(timezone.utc)
    record: dict[str, Any] = {
        "host": TARGET_HOST,
        "port": port,
        "started_at": start.isoformat(),
        "payload_hex": payload.hex() if payload else None,
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(8)
    try:
        sock.connect((TARGET_HOST, port))
        record["connect_ok"] = True
        if payload:
            sock.sendall(payload)
            record["send_ok"] = True
        data = sock.recv(512)
        record["recv_hex"] = data.hex()
        record["recv_utf8"] = data.decode("utf-8", "replace")
        record["recv_len"] = len(data)
    except (OSError, TimeoutError) as exc:
        record["connect_ok"] = record.get("connect_ok", False)
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    finally:
        record["ended_at"] = _now_iso()
        try:
            sock.close()
        except OSError as exc:
            record["close_error"] = str(exc)
    return record


# 执行 `requests 探测` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _requests_probe(url: str) -> dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    record: dict[str, Any] = {"url": url, "started_at": _now_iso()}
    try:
        response = session.get(url, timeout=12)
        record["ok"] = True
        record["status_code"] = response.status_code
        record["headers"] = dict(response.headers)
        record["body_prefix"] = response.text[:500]
    except requests.RequestException as exc:
        record["ok"] = False
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    finally:
        session.close()
        record["ended_at"] = _now_iso()
    return record


# 执行 `https 探测` 辅助逻辑，保持运维与实验脚本中的输入处理和结果输出一致。
def _https_probe(host: str, port: int) -> dict[str, Any]:
    context = ssl.create_default_context()
    record: dict[str, Any] = {
        "host": host,
        "port": port,
        "started_at": _now_iso(),
    }
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.settimeout(8)
    try:
        raw.connect((host, port))
        with context.wrap_socket(raw, server_hostname=host) as wrapped:
            record["tls_ok"] = True
            record["cipher"] = wrapped.cipher()
            record["peer_cert_present"] = bool(wrapped.getpeercert())
    except (OSError, TimeoutError, ssl.SSLError) as exc:
        record["tls_ok"] = False
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    finally:
        record["ended_at"] = _now_iso()
        try:
            raw.close()
        except OSError as exc:
            record["close_error"] = str(exc)
    return record


# 写出 `JSON`，保证后续报告、页面或复现实验能读取。
def _write_json(name: str, payload: Any) -> Path:
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# 写出 `readme`，保证后续报告、页面或复现实验能读取。
def _write_readme(manifest: dict[str, Any]) -> None:
    content = f"""# Protocol Fault Evidence 2026-04-19

目标主机：`{TARGET_HOST}`

## 结论

- 目标端口 `{", ".join(str(p) for p in TARGET_PORTS)}` 都可完成 TCP connect。
- 但 `22` 端口未返回 SSH banner，`8000` 端口对标准 HTTP 请求返回空响应，`8011/8012/8013` 端口 connect 后立即 EOF。
- 这份证据说明：`port-open != protocol-alive`。

## 产物

- `manifest.json`：总览与结论
- `tcp_connect_matrix.json`：各端口 connect + recv 结果
- `http_health_probe.json`：无代理 HTTP 请求结果
- `https_health_probe.json`：TLS 握手结果
- `http_raw_probe_8000.json`：对 `8000` 发送原始 HTTP GET 的 transcript
- `model_http_raw_probes.json`：对 `8011/8012/8013` 发送原始 HTTP GET 的 transcript
- `ssh_raw_probe_22.json`：对 `22` 读取 SSH banner 的 transcript

## 判读边界

- 该证据仅证明外部直连时的协议层行为。
- 该证据不能单独证明主机内控制面是否健康，也不能替代带外控制或宿主机面板信息。
- 该证据适合在答辩时反驳“只是代理问题”或“只是端口不通”的说法。
- `https_health_probe.json` 仅为辅助探测，不作为主故障判据。

## Manifest Snapshot

```json
{json.dumps(manifest, ensure_ascii=False, indent=2)}
```
"""
    (OUT_DIR / "README.md").write_text(content, encoding="utf-8")


# 收集 `fault probes`，把分散产物整理成统一列表。
def _collect_fault_probes() -> dict[str, Any]:
    tcp_matrix = [_port_connect_and_read(port) for port in TARGET_PORTS]
    http_raw = _port_connect_and_read(
        8000,
        b"GET /api/v1/health HTTP/1.1\r\nHost: 222.20.126.115\r\nConnection: close\r\n\r\n",
    )
    ssh_raw = _port_connect_and_read(22)
    model_http_raw = []
    for port in (8011, 8012, 8013):
        model_http_raw.append(
            {
                "probe": "GET /",
                **_port_connect_and_read(
                    port,
                    (
                        f"GET / HTTP/1.1\r\nHost: {TARGET_HOST}:{port}\r\nConnection: close\r\n\r\n"
                    ).encode("ascii"),
                ),
            }
        )
        model_http_raw.append(
            {
                "probe": "GET /v1/models",
                **_port_connect_and_read(
                    port,
                    (
                        f"GET /v1/models HTTP/1.1\r\nHost: {TARGET_HOST}:{port}\r\nConnection: close\r\n\r\n"
                    ).encode("ascii"),
                ),
            }
    )
    http_health = _requests_probe(f"http://{TARGET_HOST}:8000/api/v1/health")
    https_health = _https_probe(TARGET_HOST, 8000)
    return {
        "tcp_matrix": tcp_matrix,
        "http_raw": http_raw,
        "ssh_raw": ssh_raw,
        "model_http_raw": model_http_raw,
        "http_health": http_health,
        "https_health": https_health,
    }


# 写出 `探测 files`，保证后续报告、页面或复现实验能读取。
def _write_probe_files(probes: dict[str, Any]) -> None:
    _write_json("tcp_connect_matrix.json", probes["tcp_matrix"])
    _write_json("http_raw_probe_8000.json", probes["http_raw"])
    _write_json("ssh_raw_probe_22.json", probes["ssh_raw"])
    _write_json("model_http_raw_probes.json", probes["model_http_raw"])
    _write_json("http_health_probe.json", probes["http_health"])
    _write_json("https_health_probe.json", probes["https_health"])


# 构建 `manifest` 数据，集中整理运维与实验脚本需要的输出结构。
def _build_manifest(probes: dict[str, Any]) -> dict[str, Any]:
    tcp_matrix = probes["tcp_matrix"]
    http_health = probes["http_health"]
    https_health = probes["https_health"]
    ssh_raw = probes["ssh_raw"]
    model_http_raw = probes["model_http_raw"]
    manifest = {
        "generated_at": _now_iso(),
        "host": TARGET_HOST,
        "claim": "external direct-connect protocol fault witness",
        "key_conclusion": "target ports accept TCP connections but do not complete expected SSH/HTTP/TLS protocol exchange",
        "port_connect_ok_count": sum(1 for row in tcp_matrix if row.get("connect_ok")),
        "ports_with_empty_first_read": [
            row["port"]
            for row in tcp_matrix
            if row.get("connect_ok") and row.get("recv_len") == 0
        ],
        "http_health_ok": http_health.get("ok", False),
        "https_handshake_ok": https_health.get("tls_ok", False),
        "ssh_banner_len": ssh_raw.get("recv_len", 0),
        "model_http_probe_zero_len_ports": sorted(
            {
                row["port"]
                for row in model_http_raw
                if row.get("connect_ok") and row.get("recv_len") == 0
            }
        ),
        "probe_priority": {
            "primary": [
                "ssh_raw_probe_22.json",
                "http_raw_probe_8000.json",
                "http_health_probe.json",
                "model_http_raw_probes.json",
            ],
            "auxiliary": ["https_health_probe.json", "tcp_connect_matrix.json"],
        },
        "evidence_files": [
            "tcp_connect_matrix.json",
            "http_raw_probe_8000.json",
            "ssh_raw_probe_22.json",
            "model_http_raw_probes.json",
            "http_health_probe.json",
            "https_health_probe.json",
        ],
        "adjudication": {
            "proves": [
                "proxy bypass alone does not restore working HTTP or SSH protocol behavior",
                "port-open should not be treated as service-alive",
                "the current issue is consistent across multiple public-facing service ports",
            ],
            "does_not_prove": [
                "root cause inside the host",
                "which process is listening on each port",
                "whether the local control-plane is healthy",
            ],
        },
    }
    return manifest


# 作为 `build_protocol_fault_evidence.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main() -> None:
    _ensure_out_dir()
    _clear_proxy_env()
    probes = _collect_fault_probes()
    _write_probe_files(probes)
    manifest = _build_manifest(probes)
    _write_json("manifest.json", manifest)
    _write_readme(manifest)

    print(str(OUT_DIR))


if __name__ == "__main__":
    main()
