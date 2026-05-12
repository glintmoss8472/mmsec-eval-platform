# 文件说明：该文件属于自动化测试，集中实现 test generation metrics 相关逻辑。
from __future__ import annotations

from mmsec_eval.metrics.generation import answer_matches, normalize_answer, object_jaccard, object_present, text_similarity, yes_no_value


# 中文注释：验证 test_answer_normalization_and_alias_match 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_answer_normalization_and_alias_match() -> None:
    assert normalize_answer("The Red, Object!") == "red object"
    assert answer_matches("a round object", "circle", ["round object"])
    assert not answer_matches("square", "circle", ["round object"])


# 中文注释：验证 test_yes_no_and_object_metrics 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_yes_no_and_object_metrics() -> None:
    assert yes_no_value("Yes, there is one.") is True
    assert yes_no_value("No.") is False
    assert yes_no_value("maybe") is None
    assert object_present("A red circle is visible.", "circle", ["round object"])
    assert not object_present("A red square is visible.", "circle", ["round object"])
    assert object_jaccard(["circle", "background"], ["background"]) == 0.5


# 中文注释：验证 test_text_similarity_bounds 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_text_similarity_bounds() -> None:
    assert text_similarity("red circle", "red circle") == 1.0
    assert 0.0 <= text_similarity("red circle", "blue square") <= 1.0
