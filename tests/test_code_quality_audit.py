from __future__ import annotations

from pathlib import Path

from scripts import audit_code_quality


def test_project_code_quality_gate_is_clean():
    report = audit_code_quality._build_report(Path.cwd(), max_function_lines=80, top_n=5)

    assert report["long_function_count"] == 0
    assert report["wide_exception_count"] == 0
    assert report["silent_exception_count"] == 0
    assert report["syntax_error_count"] == 0
    assert report["python_file_count"] >= 1
