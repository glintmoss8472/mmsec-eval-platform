# 文件说明：该文件属于自动化测试，集中实现 test risk scoring 相关逻辑。
from __future__ import annotations

import pytest

from mmsec_eval.risk.components import component_catalog, component_keys
from mmsec_eval.risk.scoring import compute_risk_score


# 验证 `风险 分数 weighted sum and level` 场景，防止相关行为在后续修改中退化。
def test_risk_score_weighted_sum_and_level():
    out = compute_risk_score(
        scenario="retrieval",
        components={
            "effectiveness": 0.8,
            "semantic": 0.7,
            "cost": 0.6,
            "transfer": 0.9,
            "stability": 0.5,
        },
        weights={
            "effectiveness": 0.4,
            "semantic": 0.2,
            "cost": 0.1,
            "transfer": 0.2,
            "stability": 0.1,
        },
    )
    assert 0.0 <= float(out["risk_score"]) <= 1.0
    assert str(out["risk_level"]) in {"minimal", "low", "medium", "high", "critical"}
    assert str(out["risk_scenario"]) == "retrieval"
    rb = out["risk_breakdown"]
    assert isinstance(rb, dict)
    assert "effectiveness" in rb
    audit = out["risk_component_audit"]
    assert isinstance(audit, list)
    assert {row["key"] for row in audit} == set(component_keys())
    assert sum(float(row["contribution"]) for row in audit) == pytest.approx(float(out["risk_score"]), abs=1e-5)
    assert isinstance(out["risk_recommendations"], list)


# 验证 `风险 分数 uses scenario defaults when weights empty` 场景，防止相关行为在后续修改中退化。
def test_risk_score_uses_scenario_defaults_when_weights_empty():
    out = compute_risk_score(
        scenario="embodied",
        components={
            "effectiveness": 1.0,
            "semantic": 0.0,
            "cost": 0.0,
            "transfer": 0.0,
            "stability": 1.0,
        },
        weights={},
    )
    w = out["risk_weights"]
    assert isinstance(w, dict)
    # Embodied scenario should emphasize stability over cost.
    assert float(w["stability"]) > float(w["cost"])


# 验证 `风险 component catalog 是否 auditable` 场景，防止相关行为在后续修改中退化。
def test_risk_component_catalog_is_auditable():
    catalog = component_catalog()
    assert len(catalog) == 5
    assert {item["key"] for item in catalog} == set(component_keys())
    assert all(item["label_zh"] and item["label_en"] for item in catalog)
