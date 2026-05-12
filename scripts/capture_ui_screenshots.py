# 文件说明：该文件属于运维与实验脚本，集中实现 capture ui screenshots 相关逻辑。
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


# 中文注释：定义 PageShot 的结构化职责，作为运维与实验脚本中状态、配置或行为的边界。
@dataclass(frozen=True)
class PageShot:
    route: str
    label: str
    viewport_path: str
    full_page_path: str


DEFAULT_ROUTES: tuple[tuple[str, str], ...] = (
    ("/", "server_dashboard"),
    ("/testing", "server_testing"),
    ("/glossary", "server_glossary"),
)


# 中文注释：封装 _parse_viewport 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _parse_viewport(value: str) -> tuple[int, int]:
    raw = str(value or "1440x900").lower().replace("*", "x")
    left, _, right = raw.partition("x")
    width = int(left or "1440")
    height = int(right or "900")
    if width <= 0 or height <= 0:
        raise ValueError("viewport must be positive, for example 1440x900")
    return width, height


# 中文注释：封装 _route_url 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _route_url(base_url: str, route: str) -> str:
    return f"{base_url.rstrip('/')}/{route.lstrip('/')}"


# 中文注释：实现 capture_pages 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def capture_pages(*, base_url: str, out_dir: Path, viewport: tuple[int, int], timeout_ms: int) -> list[PageShot]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required: install frontend test dependencies or run `python -m playwright install chromium`.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    shots: list[PageShot] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        for route, label in DEFAULT_ROUTES:
            page.goto(_route_url(base_url, route), wait_until="networkidle", timeout=timeout_ms)
            viewport_path = out_dir / f"{label}_latest.png"
            full_path = out_dir / f"{label}_latest_full.png"
            page.screenshot(path=str(viewport_path), full_page=False)
            page.screenshot(path=str(full_path), full_page=True)
            shots.append(
                PageShot(
                    route=route,
                    label=label,
                    viewport_path=str(viewport_path),
                    full_page_path=str(full_path),
                )
            )
        browser.close()
    return shots


# 中文注释：实现 check_existing 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def check_existing(out_dir: Path) -> list[str]:
    required: list[str] = []
    for _, label in DEFAULT_ROUTES:
        required.append(str(out_dir / f"{label}_latest.png"))
        required.append(str(out_dir / f"{label}_latest_full.png"))
    return [path for path in required if not Path(path).exists()]


# 中文注释：实现 parse_args 的核心流程，支撑运维与实验脚本中的业务语义和异常边界。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture fixed UI screenshots used by the thesis evidence bundle.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000", help="Frontend base URL.")
    parser.add_argument("--out-dir", default="docs/assets/server_ui_content_audit_20260419", help="Screenshot output directory.")
    parser.add_argument("--viewport", default="1440x900", help="Viewport size, for example 1440x900.")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Page navigation timeout in milliseconds.")
    parser.add_argument("--check-only", action="store_true", help="Only verify that required screenshot files already exist.")
    return parser.parse_args()


# 中文注释：串联 main 的主流程，集中处理运维与实验脚本的初始化、执行和退出条件。
def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if args.check_only:
        missing = check_existing(out_dir)
        print(json.dumps({"passed": not missing, "missing": missing}, ensure_ascii=False, indent=2))
        return 0 if not missing else 1
    shots = capture_pages(
        base_url=str(args.base_url),
        out_dir=out_dir,
        viewport=_parse_viewport(args.viewport),
        timeout_ms=int(args.timeout_ms),
    )
    print(json.dumps({"passed": True, "shots": [asdict(item) for item in shots]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
