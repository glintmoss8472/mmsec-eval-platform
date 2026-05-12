# 文件说明：该文件属于自动化测试，集中实现 test code quality audit 相关逻辑。
from __future__ import annotations

from pathlib import Path

from scripts import audit_code_quality


# 验证 `project code quality gate 是否 clean` 场景，防止相关行为在后续修改中退化。
def test_project_code_quality_gate_is_clean():
    report = audit_code_quality._build_report(Path.cwd(), max_function_lines=80, top_n=5)

    assert report["long_function_count"] == 0
    assert report["wide_exception_count"] == 0
    assert report["silent_exception_count"] == 0
    assert report["syntax_error_count"] == 0
    assert report["python_file_count"] >= 1
