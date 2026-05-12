# 文件说明：该文件属于自动化测试，集中实现 test code quality audit 相关逻辑。
from __future__ import annotations

from pathlib import Path

from scripts import audit_code_quality


# 中文注释：验证 test_project_code_quality_gate_is_clean 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_project_code_quality_gate_is_clean():
    report = audit_code_quality._build_report(Path.cwd(), max_function_lines=80, top_n=5)

    assert report["long_function_count"] == 0
    assert report["wide_exception_count"] == 0
    assert report["silent_exception_count"] == 0
    assert report["syntax_error_count"] == 0
    assert report["python_file_count"] >= 1
