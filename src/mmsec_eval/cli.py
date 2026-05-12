# 文件说明：该文件属于项目工程，集中实现 cli 相关逻辑。
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mmsec_eval.config.loader import load_config
from mmsec_eval.config.sweep import apply_override
from mmsec_eval.config.validate import validate_config
from mmsec_eval.docs_ingest.local_sources import local_source_to_dict, resolve_local_sources
from mmsec_eval.docs_ingest.parsers import parse_doc, parse_pdf, parse_text
from mmsec_eval.docs_ingest.summarizer import make_snippets
from mmsec_eval.io.jsonl_io import read_jsonl, write_jsonl
from mmsec_eval.logging import setup_logging
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.runtime import apply_config_env


# 作为 `cli.py` 的执行入口，串联参数读取、核心处理和退出状态。
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mmsec_eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest-docs")
    p_ing.add_argument("--config", required=True)

    p_run = sub.add_parser("run-eval")
    p_run.add_argument("--config", required=True)

    p_vlr = sub.add_parser("run-vlr")
    p_vlr.add_argument("--config", required=True)

    p_train_advclip = sub.add_parser("train-advclip")
    p_train_advclip.add_argument("--config", required=True)

    p_bench = sub.add_parser("run-benchmark")
    p_bench.add_argument("--config", required=True)

    p_sweep = sub.add_parser("run-sweep")
    p_sweep.add_argument("--config", required=True)
    p_sweep.add_argument("--sweep", required=False, default="")

    args = parser.parse_args(argv)

    if args.cmd == "ingest-docs":
        return cmd_ingest_docs(args.config)
    if args.cmd == "run-eval":
        return cmd_run_eval(args.config)
    if args.cmd == "run-vlr":
        return cmd_run_vlr(args.config)
    if args.cmd == "train-advclip":
        return cmd_train_advclip(args.config)
    if args.cmd == "run-benchmark":
        return cmd_run_benchmark(args.config)
    if args.cmd == "run-sweep":
        return cmd_run_sweep(args.config, args.sweep)
    return 1


# 执行 `echo` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def _echo(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write((msg + "\n").encode(enc, errors="replace"))


# 执行 `cmd ingest 文档` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def cmd_ingest_docs(config_path: str) -> int:
    register_builtin_plugins()
    cfg = load_config(config_path)
    _apply_runtime_env(cfg)
    validate_config(cfg)
    setup_logging(cfg.artifacts_dir)
    sources = resolve_local_sources(cfg)

    docs_index_path = Path(cfg.artifacts_dir) / "docs_index.json"
    snippets_path = Path(cfg.artifacts_dir) / "docs_snippets.jsonl"
    docs_index_path.parent.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    snippet_rows: list[dict[str, Any]] = []
    parsed_count = 0

    for src in sources:
        row = local_source_to_dict(src)
        text = ""
        if src.exists:
            if src.parser == "parse_pdf":
                text = parse_pdf(src.resolved_path, max_pages=cfg.docs.max_pages)
            elif src.parser == "parse_doc":
                text = parse_doc(src.resolved_path)
            else:
                text = parse_text(src.resolved_path)
            snip = make_snippets(text, max_chars=cfg.docs.snippet_chars)
            parsed_count += 1 if snip["length"] > 0 else 0
            row["summary_200"] = snip["first_200"]
            row["parse_quality"] = "ok" if snip["length"] > 0 else "empty"
            snippet_rows.append(
                {
                    "requested_path": src.requested_path,
                    "resolved_path": src.resolved_path,
                    "snippet": snip["snippet"],
                    "first_200": snip["first_200"],
                    "length": snip["length"],
                }
            )
            _echo(f"[DOC] {Path(src.resolved_path).name}: {snip['first_200']}")
        else:
            row["parse_quality"] = "missing"
        index_rows.append(row)

    docs_index_path.write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(str(snippets_path), snippet_rows)
    _echo(f"[DOC] index={docs_index_path}")
    _echo(f"[DOC] snippets={snippets_path}")
    _echo(f"[DOC] parsed_count={parsed_count}")
    return 0


# 应用 `runtime 环境` 规则，把兼容字段写回报告或风险载荷。
def _apply_runtime_env(cfg) -> None:
    apply_config_env(cfg)


# 执行 `评测 internal` 流程，按配置驱动项目工程完成一次任务。
def _run_eval_internal(
    config_path: str,
    sweep_override: dict[str, Any] | None = None,
    benchmark_mode: bool = False,
):
    from mmsec_eval.runner.eval_runner import run

    cfg = load_config(config_path)
    if sweep_override:
        cfg = apply_override(cfg, sweep_override)
    _apply_runtime_env(cfg)
    validate_config(cfg)
    setup_logging(cfg.artifacts_dir)
    artifacts = run(cfg, benchmark_mode=benchmark_mode)
    return cfg, artifacts


# 执行 `cmd 运行记录 评测` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def cmd_run_eval(config_path: str, sweep_override: dict[str, Any] | None = None) -> int:
    register_builtin_plugins()
    cfg, artifacts = _run_eval_internal(config_path, sweep_override=sweep_override, benchmark_mode=False)
    _echo(f"[RUN] run_id={artifacts.run_id}")
    _echo(f"[RUN] results={artifacts.results_path}")
    _echo(f"[RUN] summary={artifacts.summary_path}")
    _echo(f"[RUN] report={artifacts.report_path}")
    _echo(f"[RUN] dataset={cfg.dataset.kind}")
    if artifacts.run_index_path:
        _echo(f"[RUN] cases_index={artifacts.run_index_path}")
    if artifacts.benchmark_summary_path:
        _echo(f"[RUN] benchmark_summary={artifacts.benchmark_summary_path}")
    return 0


# 执行 `cmd 运行记录 图文检索` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def cmd_run_vlr(config_path: str) -> int:
    register_builtin_plugins()
    from mmsec_eval.runner.retrieval_runner import run as run_vlr

    cfg = load_config(config_path)
    # Enforce VLR semantics for this CLI entrypoint.
    cfg.task.kind = "vlr"
    _apply_runtime_env(cfg)
    validate_config(cfg)
    setup_logging(cfg.artifacts_dir)
    artifacts = run_vlr(cfg)

    _echo(f"[VLR] run_id={artifacts.run_id}")
    _echo(f"[VLR] results={artifacts.results_path}")
    _echo(f"[VLR] summary={artifacts.summary_path}")
    _echo(f"[VLR] report={artifacts.report_path}")
    _echo(f"[VLR] dataset={cfg.dataset.kind}")
    return 0


# 执行 `cmd train advclip` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def cmd_train_advclip(config_path: str) -> int:
    """Train an AdvCLIP universal patch and save it into the run directory."""
    register_builtin_plugins()
    from mmsec_eval.attacks.advclip.train import train_advclip_patch

    cfg = load_config(config_path)
    # Training runs in strict real mode (CUDA-only, no fallback execution paths).
    cfg.model.enable_gradients = True
    cfg.task.kind = "vlr"
    cfg.plugins.attack = "advclip"
    _apply_runtime_env(cfg)
    validate_config(cfg)
    setup_logging(cfg.artifacts_dir)

    artifacts = train_advclip_patch(cfg)
    _echo(f"[ADVCLIP] run_id={artifacts.run_id}")
    _echo(f"[ADVCLIP] results={artifacts.results_path}")
    _echo(f"[ADVCLIP] summary={artifacts.summary_path}")
    _echo(f"[ADVCLIP] report={artifacts.report_path}")
    return 0


# 执行 `cmd 运行记录 基准评测` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def cmd_run_benchmark(config_path: str) -> int:
    register_builtin_plugins()
    cfg, artifacts = _run_eval_internal(config_path, sweep_override=None, benchmark_mode=True)
    _echo(f"[BENCH] run_id={artifacts.run_id}")
    _echo(f"[BENCH] dataset={cfg.dataset.kind}")
    _echo(f"[BENCH] results={artifacts.results_path}")
    _echo(f"[BENCH] summary={artifacts.summary_path}")
    if artifacts.benchmark_summary_path:
        _echo(f"[BENCH] benchmark_summary={artifacts.benchmark_summary_path}")
    return 0


# 执行 `cmd 运行记录 参数扫描` 辅助逻辑，保持项目工程中的输入处理和结果输出一致。
def cmd_run_sweep(config_path: str, sweep_path: str = "") -> int:
    register_builtin_plugins()
    base_cfg = load_config(config_path)
    _apply_runtime_env(base_cfg)
    validate_config(base_cfg)
    sweep_file = sweep_path or base_cfg.sweep.path
    overrides = read_jsonl(sweep_file)
    if not overrides:
        _echo(f"[SWEEP] no overrides found: {sweep_file}")
        return 1

    run_index_rows: list[dict[str, Any]] = []
    for idx, ov in enumerate(overrides):
        _echo(f"[SWEEP] running override #{idx}")
        cfg, artifacts = _run_eval_internal(config_path, sweep_override=ov, benchmark_mode=False)

        summary = {}
        try:
            summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            summary = {}

        run_index_rows.append(
            {
                "summary_path": artifacts.summary_path,
                "results_path": artifacts.results_path,
                "report_path": artifacts.report_path,
                "override": ov,
                "run_id": artifacts.run_id,
                "asr": summary.get("asr", 0.0),
                "avg_l2": summary.get("avg_l2", 0.0),
                "dataset_name": cfg.dataset.kind,
                "benchmark_tag": cfg.dataset.benchmark_tag or cfg.dataset.kind,
            }
        )

    run_index_path = Path(base_cfg.artifacts_dir) / "runs" / "run_index.jsonl"
    write_jsonl(str(run_index_path), run_index_rows)
    _echo(f"[SWEEP] run_index={run_index_path}")
    return 0
