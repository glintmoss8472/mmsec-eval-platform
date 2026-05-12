# 文件说明：该文件属于项目工程，集中实现 main 相关逻辑。
from __future__ import annotations

import argparse

import uvicorn


# 中文注释：封装 _parse_args 的内部步骤，让项目工程主流程保持清晰并隔离边界细节。
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ATT-project FastAPI service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run("mmsec_api.main:app", host=args.host, port=args.port, reload=bool(args.reload))
