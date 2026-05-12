# 文件说明：该文件属于风险评分层，集中实现 init 相关逻辑。
from __future__ import annotations

from mmsec_eval.risk.components import component_catalog, component_keys
from mmsec_eval.risk.scoring import compute_risk_score

__all__ = ["component_catalog", "component_keys", "compute_risk_score"]
