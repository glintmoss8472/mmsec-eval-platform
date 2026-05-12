# 文件说明：该文件属于运维与实验脚本，集中实现 audit code quality 相关逻辑。
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


EXCLUDED_PREFIXES = (
    ".venv/",
    "vendor/",
    "third_party/",
    "external/",
    "frontend/dist/",
    "AdvCLIP/",
    "TMM-main/",
    "16785_AdvEDM_Fine_grained_Adve_Supplementary Material/",
)
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".md"}
LONG_FUNCTION_ALLOWLIST = {
    ("src/mmsec_api/store/sqlite.py", "upsert_sample_assets"),
    ("src/mmsec_api/store/sqlite.py", "list_sample_assets"),
    ("src/mmsec_api/store/sqlite.py", "list_sample_asset_batches"),
    ("src/mmsec_api/services/asset_evaluator.py", "run_asset_evaluation"),
}


# 中文注释：封装 _git_files 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _git_files(root: Path, pattern: str | None = None) -> list[str]:
    cmd = ["git", "ls-files"]
    if pattern:
        cmd.append(pattern)
    return subprocess.check_output(cmd, cwd=root, text=True).splitlines()


# 中文注释：封装 _is_project_file 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _is_project_file(rel_path: str) -> bool:
    return not rel_path.startswith(EXCLUDED_PREFIXES)


# 中文注释：封装 _read_text 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return None


# 中文注释：封装 _scan_python_file 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _scan_python_file(root: Path, rel_path: str, *, max_function_lines: int) -> dict[str, list[dict[str, Any]]]:
    path = root / rel_path
    text = _read_text(path)
    if text is None:
        return {"long_functions": [], "wide_exceptions": [], "silent_exceptions": [], "syntax_errors": []}
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "long_functions": [],
            "wide_exceptions": [],
            "silent_exceptions": [],
            "syntax_errors": [{"file": rel_path, "line": exc.lineno, "message": exc.msg}],
        }

    long_functions: list[dict[str, Any]] = []
    wide_exceptions: list[dict[str, Any]] = []
    silent_exceptions: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = int(getattr(node, "end_lineno", node.lineno) - node.lineno + 1)
            if length >= max_function_lines and (rel_path, node.name) not in LONG_FUNCTION_ALLOWLIST:
                long_functions.append({"file": rel_path, "line": node.lineno, "name": node.name, "lines": length})
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                wide_exceptions.append({"file": rel_path, "line": node.lineno, "kind": "bare"})
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                wide_exceptions.append({"file": rel_path, "line": node.lineno, "kind": "Exception"})
            if _except_body_is_silent(node.body):
                silent_exceptions.append({"file": rel_path, "line": node.lineno})
    return {
        "long_functions": long_functions,
        "wide_exceptions": wide_exceptions,
        "silent_exceptions": silent_exceptions,
        "syntax_errors": [],
    }


# 中文注释：封装 _except_body_is_silent 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _except_body_is_silent(body: list[ast.stmt]) -> bool:
    statements = [stmt for stmt in body if not _is_docstring_expr(stmt)]
    return bool(statements) and all(isinstance(stmt, ast.Pass) for stmt in statements)


# 中文注释：封装 _is_docstring_expr 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _is_docstring_expr(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant) and isinstance(stmt.value.value, str)


# 中文注释：封装 _python_quality 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _python_quality(root: Path, *, max_function_lines: int) -> dict[str, Any]:
    long_functions: list[dict[str, Any]] = []
    wide_exceptions: list[dict[str, Any]] = []
    silent_exceptions: list[dict[str, Any]] = []
    syntax_errors: list[dict[str, Any]] = []
    python_files = [rel for rel in _git_files(root, "*.py") if _is_project_file(rel)]
    for rel_path in python_files:
        result = _scan_python_file(root, rel_path, max_function_lines=max_function_lines)
        long_functions.extend(result["long_functions"])
        wide_exceptions.extend(result["wide_exceptions"])
        silent_exceptions.extend(result["silent_exceptions"])
        syntax_errors.extend(result["syntax_errors"])
    long_functions.sort(key=lambda row: int(row["lines"]), reverse=True)
    return {
        "python_file_count": len(python_files),
        "long_functions": long_functions,
        "wide_exceptions": wide_exceptions,
        "silent_exceptions": silent_exceptions,
        "syntax_errors": syntax_errors,
    }


# 中文注释：封装 _line_count_summary 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _line_count_summary(root: Path, *, top_n: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rel_path in _git_files(root):
        if not _is_project_file(rel_path):
            continue
        path = root / rel_path
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        text = _read_text(path)
        if text is None:
            continue
        nonblank = sum(1 for line in text.splitlines() if line.strip())
        rows.append({"file": rel_path, "nonblank_lines": nonblank})
    rows.sort(key=lambda row: int(row["nonblank_lines"]), reverse=True)
    return {"tracked_source_doc_files": len(rows), "nonblank_total": sum(int(row["nonblank_lines"]) for row in rows), "largest_files": rows[:top_n]}


# 中文注释：封装 _build_report 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _build_report(root: Path, *, max_function_lines: int, top_n: int) -> dict[str, Any]:
    quality = _python_quality(root, max_function_lines=max_function_lines)
    line_counts = _line_count_summary(root, top_n=top_n)
    return {
        "project_root": str(root),
        "max_function_lines": max_function_lines,
        "python_file_count": quality["python_file_count"],
        "long_function_count": len(quality["long_functions"]),
        "wide_exception_count": len(quality["wide_exceptions"]),
        "silent_exception_count": len(quality["silent_exceptions"]),
        "syntax_error_count": len(quality["syntax_errors"]),
        "long_functions": quality["long_functions"],
        "wide_exceptions": quality["wide_exceptions"],
        "silent_exceptions": quality["silent_exceptions"],
        "syntax_errors": quality["syntax_errors"],
        **line_counts,
    }


# 中文注释：封装 _print_text 的内部步骤，让运维与实验脚本主流程保持清晰并隔离边界细节。
def _print_text(report: dict[str, Any]) -> None:
    print(f"project_root: {report['project_root']}")
    print(f"python_file_count: {report['python_file_count']}")
    print(f"long_function_count: {report['long_function_count']}")
    print(f"wide_exception_count: {report['wide_exception_count']}")
    print(f"silent_exception_count: {report['silent_exception_count']}")
    print(f"syntax_error_count: {report['syntax_error_count']}")
    print(f"nonblank_total: {report['nonblank_total']}")
    print("largest_files:")
    for row in report["largest_files"]:
        print(f"  {row['nonblank_lines']:5d} {row['file']}")
    if report["long_functions"]:
        print("long_functions:")
        for row in report["long_functions"]:
            print(f"  {row['lines']:4d} {row['file']}:{row['line']} {row['name']}")


# 中文注释：串联 main 的主流程，集中处理运维与实验脚本的初始化、执行和退出条件。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--max-function-lines", type=int, default=80)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    report = _build_report(
        Path(args.root).resolve(),
        max_function_lines=max(1, int(args.max_function_lines)),
        top_n=max(1, int(args.top_n)),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_text(report)
    has_issues = bool(
        report["long_function_count"]
        or report["wide_exception_count"]
        or report["silent_exception_count"]
        or report["syntax_error_count"]
    )
    return 1 if args.fail_on_issues and has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
