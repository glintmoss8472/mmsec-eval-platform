# 文件说明：该文件属于自动化测试，集中实现 test vqa dialogue benchmark 相关逻辑。
from __future__ import annotations

from mmsec_eval.interaction.vqa_dialogue_benchmark import evaluate_interaction_cases, summarize_interaction_cases


# 验证 `视觉问答 dialogue counts clean correct and 攻击后样本 wrong` 场景，防止相关行为在后续修改中退化。
def test_vqa_dialogue_counts_clean_correct_and_attacked_wrong():
    rows = [
        {
            "case_id": "case_1",
            "case_type": "vqa",
            "acceptable_answers": ["red", "红色"],
            "wrong_answers": ["blue", "蓝色"],
            "clean_output": '{"answer": "红色", "reason": "visible target"}',
            "attacked_output": '{"answer": "蓝色", "reason": "perturbed target"}',
            "attack_name": "tmm",
            "semantic_preserved": True,
        }
    ]
    results = evaluate_interaction_cases(rows)
    summary = summarize_interaction_cases(results)
    assert results[0]["clean_correct"] is True
    assert results[0]["attacked_wrong"] is True
    assert summary["clean_correct_rate"] == 1.0
    assert summary["case_type_counts"]["vqa"] == 1
