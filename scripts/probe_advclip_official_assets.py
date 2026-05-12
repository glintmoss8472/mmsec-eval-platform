from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UTC = timezone.utc
DEFAULT_SHARE_ID = "JqKbqGfTRs"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_cstcloud_share(share_id: str, *, timeout: int) -> dict[str, Any]:
    api_url = f"https://pan.cstcloud.cn/s/api/shareGetInfo?shareId={share_id}"
    page_url = f"https://pan.cstcloud.cn/s/{share_id}"
    result: dict[str, Any] = {
        "name": "advclip_cstcloud_raw_data",
        "share_id": share_id,
        "page_url": page_url,
        "api_url": api_url,
        "required_payload": "raw_data.rar containing data/<dataset>/train.pkl and data/<dataset>/test.pkl for eight AdvCLIP datasets",
    }
    try:
        request = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(2_000_000).decode("utf-8", "replace")
        result["http_status"] = int(response.status)
        result["raw_response"] = body
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"parse_error": "response is not JSON"}
        result["parsed_response"] = parsed
        stat = str(parsed.get("stat", ""))
        result["status"] = "available" if stat and stat != "ERR_SHARE_EXPIRED" else "expired"
        if stat == "ERR_SHARE_EXPIRED":
            result["blocker"] = "Official AdvCLIP CSTCloud share is expired; strict reproduction needs the original raw_data.rar or a verified replacement mirror."
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the official AdvCLIP CSTCloud dataset share without downloading local files.")
    parser.add_argument("--share-id", default=DEFAULT_SHARE_ID)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else PROJECT_ROOT / "artifacts" / "advclip_official_asset_probe"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "probe": probe_cstcloud_share(args.share_id, timeout=args.timeout),
    }
    _write_json(out_dir / "advclip_official_asset_probe.json", payload)
    print(json.dumps({"out_dir": str(out_dir), "status": payload["probe"].get("status")}, ensure_ascii=False))
    return 0 if payload["probe"].get("status") == "available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
