# 文件说明：该文件属于资料摄取层，集中实现 summarizer 相关逻辑。
from __future__ import annotations

import re


# 构建 `snippets` 数据，集中整理资料摄取层需要的输出结构。
def make_snippets(text: str, max_chars: int = 800) -> dict:
    clean = re.sub(r"\s+", " ", text or "").strip()
    head = clean[:max_chars]
    first_200 = clean[:200]
    return {
        "first_200": first_200,
        "snippet": head,
        "length": len(clean),
    }

