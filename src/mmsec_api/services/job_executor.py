# 文件说明：该文件属于后端业务服务，集中实现 job executor 相关逻辑。
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import os

from mmsec_api.services.asset_evaluator import run_asset_evaluation
from mmsec_api.services.model_runtime import ensure_models_ready, model_supports_task
from mmsec_api.services.sample_generator import run_sample_generation_only
from mmsec_api.services.sample_assets import sync_sample_assets_from_runs
from mmsec_api.store.sqlite import SQLiteStore
from mmsec_eval.cli import cmd_ingest_docs, cmd_run_sweep
from mmsec_eval.attacks.advclip.registry import make_key, resolve_patch
from mmsec_eval.attacks.catalog import attack_surrogate_error
from mmsec_eval.config.loader import load_config
from mmsec_eval.config.sweep import apply_override
from mmsec_eval.config.validate import validate_config
from mmsec_eval.logging import setup_logging
from mmsec_eval.model_adapters.local_vlm_lifecycle import local_vlm_adapters
from mmsec_eval.plugins.builtin import register_builtin_plugins
from mmsec_eval.runtime import apply_config_env
from mmsec_eval.runner.eval_runner import run


# 中文注释：定义 JobExecutor 的结构化职责，作为后端业务服务中状态、配置或行为的边界。
class JobExecutor:
    # 中文注释：封装 JobExecutor.__init__ 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def __init__(self, store: SQLiteStore, artifacts_dir: str = "artifacts") -> None:
        self.store = store
        self.artifacts_dir = artifacts_dir

    # 中文注释：封装 JobExecutor._normalize_override 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _normalize_override(override: dict[str, Any]) -> dict[str, Any]:
        runner = override.get("runner")
        if isinstance(runner, dict):
            singular_victim = runner.pop("victim_model_adapter", None)
            if singular_victim and not runner.get("victim_model_adapters"):
                runner["victim_model_adapters"] = [str(singular_victim)]
        return override

    # 中文注释：封装 JobExecutor._parse_override 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _parse_override(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return JobExecutor._normalize_override(parsed) if isinstance(parsed, dict) else {}

    # 中文注释：封装 JobExecutor._parse_payload 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _parse_payload(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # 中文注释：封装 JobExecutor._deep_merge 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = JobExecutor._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    # 中文注释：封装 JobExecutor._validate_vlr_attack_capability 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _validate_vlr_attack_capability(cfg) -> None:
        attack = str(getattr(cfg.plugins, "attack", "") or "").strip()
        surrogate = str(getattr(cfg.runner, "surrogate_model_adapter", "") or getattr(cfg.plugins, "model_adapter", "") or "").strip()
        error = attack_surrogate_error(attack, surrogate)
        if error:
            raise ValueError(error)

    # 中文注释：封装 JobExecutor._validate_task_model_capability 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _validate_task_model_capability(cfg, task_kind: str) -> None:
        task = str(task_kind or "").strip()
        if task in {"vqa", "caption"}:
            adapter = str(getattr(cfg.plugins, "model_adapter", "") or "").strip()
            if not model_supports_task(adapter, task):
                raise ValueError(
                    f"模型 {adapter} 不支持 {task.upper()} 生成式真实测评。"
                    "请使用真实视觉语言生成模型；内置演示模型、CLIP、BLIP、ViLT 只能用于开发 smoke 或检索任务，不能用于正式 VQA/Caption。"
                )
            return
        if task == "vlr":
            victim_adapters = [str(item) for item in list(getattr(cfg.runner, "victim_model_adapters", []) or []) if str(item).strip()]
            if not victim_adapters:
                victim_adapters = [str(getattr(cfg.plugins, "model_adapter", "") or "").strip()]
            blocked = [adapter for adapter in victim_adapters if not model_supports_task(adapter, "vlr")]
            if blocked:
                raise ValueError(
                    "以下模型不支持 VLR 图文检索真实测评："
                    + ", ".join(blocked)
                    + "。内置演示模型不允许作为正式受测模型。"
                )

    # 中文注释：封装 JobExecutor._apply_runtime_env 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _apply_runtime_env(self, cfg) -> None:
        # API jobs and CLI runs share the same runtime/model env semantics.
        apply_config_env(cfg)
        os.environ.setdefault("MMSEC_MODEL_PREFLIGHT_TIMEOUT_SECONDS", "1200")

    # 中文注释：封装 JobExecutor._skip_model_preflight 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _skip_model_preflight() -> bool:
        return str(os.getenv("MMSEC_SKIP_MODEL_PREFLIGHT", "0")).strip().lower() in {"1", "true", "yes", "on"}

    # 中文注释：封装 JobExecutor._run_eval_like 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _run_eval_like(
        self,
        *,
        config_path: str,
        override: dict[str, Any],
        benchmark_mode: bool,
        log: Callable[[str, str], None],
        progress: Callable[[str, str, float | None, str], None],
        force_task_kind: str | None = None,
    ) -> dict[str, Any]:
        register_builtin_plugins()
        cfg = load_config(config_path)
        if override:
            cfg = apply_override(cfg, override)
        if force_task_kind:
            cfg.task.kind = str(force_task_kind)
        cfg.defense.enabled = False
        cfg.artifacts_dir = self.artifacts_dir
        self._apply_runtime_env(cfg)
        progress("config_validation", "running", 18, "正在读取配置并应用覆盖参数。")
        validate_config(cfg)
        progress("config_validation", "success", 26, "配置校验完成。")
        if not self._skip_model_preflight():
            progress("model_preflight", "running", 12, "正在检查代理模型与受测模型是否可用。")
            ensure_models_ready(
                [str(cfg.plugins.model_adapter)],
                project_root=Path(__file__).resolve().parents[3],
                log=log,
            )
            progress("model_preflight", "success", 16, "模型预检查完成。")
        setup_logging(cfg.artifacts_dir)

        log("info", f"run start: dataset={cfg.dataset.kind} attack={cfg.plugins.attack} mode={cfg.attack.mode}")
        artifacts = run(cfg, benchmark_mode=benchmark_mode, progress=progress)
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        self.store.upsert_run_cache(summary, artifacts.run_dir)
        sync_sample_assets_from_runs(self.artifacts_dir, self.store, run_ids=[str(artifacts.run_id)])
        log("info", f"run success: run_id={artifacts.run_id}")
        return {
            "run_id": artifacts.run_id,
            "summary_path": artifacts.summary_path,
            "results_path": artifacts.results_path,
            "report_path": artifacts.report_path,
        }

    # 中文注释：封装 JobExecutor._run_vlr_like 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _run_vlr_like(
        self,
        *,
        config_path: str,
        override: dict[str, Any],
        log: Callable[[str, str], None],
        progress: Callable[[str, str, float | None, str], None],
    ) -> dict[str, Any]:
        from mmsec_eval.runner.retrieval_runner import run as run_vlr

        register_builtin_plugins()
        cfg = load_config(config_path)
        if override:
            cfg = apply_override(cfg, override)
        cfg.artifacts_dir = self.artifacts_dir
        # Enforce VLR semantics for this job type.
        cfg.task.kind = "vlr"
        cfg.defense.enabled = False
        self._apply_runtime_env(cfg)
        self._validate_vlr_attack_capability(cfg)
        self._validate_task_model_capability(cfg, "vlr")
        progress("config_validation", "running", 18, "正在读取配置并应用覆盖参数。")
        validate_config(cfg)
        progress("config_validation", "success", 26, "配置校验完成。")
        if not self._skip_model_preflight():
            progress("model_preflight", "running", 12, "正在检查代理模型与受测模型是否可用。")
            selected_models = [str(cfg.runner.surrogate_model_adapter or cfg.plugins.model_adapter)]
            selected_models.extend([str(item) for item in list(cfg.runner.victim_model_adapters or [])])
            ensure_models_ready(
                selected_models,
                project_root=Path(__file__).resolve().parents[3],
                log=log,
            )
            progress("model_preflight", "success", 16, "模型预检查完成。")
        setup_logging(cfg.artifacts_dir)

        log("info", f"run-vlr start: dataset={cfg.dataset.kind} attack={cfg.plugins.attack} scope={cfg.task.eval_scope}")

        # AdvCLIP evaluation needs a ready universal patch. The API path hides
        # that extra step by training once on demand when the registry is empty.
        if str(cfg.plugins.attack) == "advclip" and str(cfg.task.eval_scope or "clean") != "clean":
            reg_key = make_key(
                clip_model_name=str(cfg.model.clip_model_name),
                mode=str(cfg.attack.mode),
                patch_size=int(cfg.attack.patch_size),
            )
            resolved = resolve_patch(str(cfg.artifacts_dir), reg_key)
            if not resolved:
                log(
                    "info",
                    "advclip patch missing in registry; auto-running train-advclip before run-vlr",
                )
                self._train_advclip_like(config_path=config_path, override=override, log=log, progress=progress)
                resolved = resolve_patch(str(cfg.artifacts_dir), reg_key)
                if not resolved:
                    raise RuntimeError(
                        "AdvCLIP patch is still missing after auto training. "
                        "Check train-advclip logs and registry file."
                    )

        artifacts = run_vlr(cfg, progress=progress)
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        self.store.upsert_run_cache(summary, artifacts.run_dir)
        sync_sample_assets_from_runs(self.artifacts_dir, self.store, run_ids=[str(artifacts.run_id)])
        log("info", f"run-vlr success: run_id={artifacts.run_id}")
        return {
            "run_id": artifacts.run_id,
            "summary_path": artifacts.summary_path,
            "results_path": artifacts.results_path,
            "report_path": artifacts.report_path,
        }

    # 中文注释：封装 JobExecutor._run_generation_like 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _run_generation_like(
        self,
        *,
        config_path: str,
        override: dict[str, Any],
        log: Callable[[str, str], None],
        progress: Callable[[str, str, float | None, str], None],
        force_task_kind: str,
    ) -> dict[str, Any]:
        from mmsec_eval.runner.generation_runner import run as run_generation

        register_builtin_plugins()
        cfg = load_config(config_path)
        if override:
            cfg = apply_override(cfg, override)
        cfg.artifacts_dir = self.artifacts_dir
        cfg.task.kind = str(force_task_kind)
        cfg.task.eval_scope = "image"
        cfg.defense.enabled = False
        if not str(cfg.runner.surrogate_model_adapter or "").strip():
            cfg.runner.surrogate_model_adapter = "clip_hf"
        self._apply_runtime_env(cfg)
        self._validate_vlr_attack_capability(cfg)
        self._validate_task_model_capability(cfg, force_task_kind)
        progress("config_validation", "running", 18, "正在读取生成式评测配置并应用覆盖参数。")
        validate_config(cfg)
        progress("config_validation", "success", 26, "生成式评测配置校验完成。")
        if not self._skip_model_preflight():
            progress("model_preflight", "running", 12, "正在检查生成模型与攻击代理模型是否可用。")
            generation_adapter = str(cfg.plugins.model_adapter)
            surrogate_adapter = str(cfg.runner.surrogate_model_adapter or "clip_hf")
            staged_lifecycle = bool(getattr(cfg.runner, "staged_model_lifecycle", True))
            defer_generation_model = bool(staged_lifecycle and local_vlm_adapters([generation_adapter]))
            selected_models = [surrogate_adapter] if defer_generation_model else [generation_adapter, surrogate_adapter]
            if defer_generation_model:
                log("info", f"defer local VLM startup until post-attack evaluation: adapter={generation_adapter}")
            ensure_models_ready(
                list(dict.fromkeys(selected_models)),
                project_root=Path(__file__).resolve().parents[3],
                log=log,
            )
            progress("model_preflight", "success", 16, "生成模型与代理模型预检查完成。")
        setup_logging(cfg.artifacts_dir)

        log(
            "info",
            f"run-generation start: task={cfg.task.kind} dataset={cfg.dataset.kind} "
            f"benchmark={cfg.dataset.benchmark_tag or cfg.task.cases_jsonl} "
            f"attack={cfg.plugins.attack}",
        )
        artifacts = run_generation(cfg, progress=progress)
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        self.store.upsert_run_cache(summary, artifacts.run_dir)
        sync_sample_assets_from_runs(self.artifacts_dir, self.store, run_ids=[str(artifacts.run_id)])
        log("info", f"run-generation success: run_id={artifacts.run_id}")
        return {
            "run_id": artifacts.run_id,
            "summary_path": artifacts.summary_path,
            "results_path": artifacts.results_path,
            "report_path": artifacts.report_path,
        }

    # 中文注释：封装 JobExecutor._train_advclip_like 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _train_advclip_like(
        self,
        *,
        config_path: str,
        override: dict[str, Any],
        log: Callable[[str, str], None],
        progress: Callable[[str, str, float | None, str], None],
    ) -> dict[str, Any]:
        from mmsec_eval.attacks.advclip.train import train_advclip_patch

        register_builtin_plugins()
        cfg = load_config(config_path)
        if override:
            cfg = apply_override(cfg, override)
        cfg.artifacts_dir = self.artifacts_dir
        # AdvCLIP training runs are defined for VLR/CLIP surrogate semantics.
        cfg.task.kind = "vlr"
        cfg.plugins.attack = "advclip"
        self._apply_runtime_env(cfg)
        progress("config_validation", "running", 18, "正在读取对抗补丁训练配置。")
        validate_config(cfg)
        progress("config_validation", "success", 28, "训练配置校验完成。")
        if not self._skip_model_preflight():
            progress("model_preflight", "running", 12, "正在检查补丁训练所需模型。")
            ensure_models_ready(
                [str(cfg.runner.surrogate_model_adapter or cfg.plugins.model_adapter)],
                project_root=Path(__file__).resolve().parents[3],
                log=log,
            )
            progress("model_preflight", "success", 18, "补丁训练模型检查完成。")
        setup_logging(cfg.artifacts_dir)

        log(
            "info",
            f"train-advclip start: surrogate={str(cfg.runner.surrogate_model_adapter or cfg.plugins.model_adapter)} "
            f"mode={str(cfg.attack.mode)} patch={int(cfg.attack.patch_size)} use_gan={1 if bool(cfg.attack.use_gan) else 0} "
            f"steps={int(cfg.attack.patch_train_steps or 0)}",
        )
        progress("dataset_loading", "running", 38, "正在准备补丁训练数据。")
        progress("attack_execution", "running", 62, "正在执行对抗补丁训练。")
        artifacts = train_advclip_patch(cfg)
        progress("result_aggregation", "running", 90, "正在汇总补丁训练结果。")
        progress("report_writing", "running", 97, "正在写入补丁训练报告。")
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        self.store.upsert_run_cache(summary, artifacts.run_dir)
        log("info", f"train-advclip success: run_id={artifacts.run_id}")
        return {
            "run_id": artifacts.run_id,
            "summary_path": artifacts.summary_path,
            "results_path": artifacts.results_path,
            "report_path": artifacts.report_path,
        }

    # 中文注释：封装 JobExecutor._dataset_root_path 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _dataset_root_path(name: str, payload: dict[str, Any]) -> str:
        root_path = str(payload.get("root_path", "") or "").strip()
        if name == "flickr30k":
            if not root_path or Path(root_path).name.lower() == "coco":
                root_path = "data/flickr30k"
        elif name == "flickr1k":
            if not root_path or Path(root_path).name.lower() in {"coco", "flickr30k"}:
                root_path = "data/flickr30k"
        elif name == "mini_flickr":
            if not root_path:
                root_path = "data/mini_flickr"
        else:
            if not root_path or Path(root_path).name.lower() == "flickr30k":
                root_path = "data/coco"
        return root_path

    # 中文注释：封装 JobExecutor._script_cmd 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _script_cmd(project_root: Path, is_windows: bool, ps1_name: str, py_name: str) -> list[str]:
        if is_windows:
            script = project_root / "scripts" / ps1_name
            return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        return [sys.executable, str(project_root / "scripts" / py_name)]

    # 中文注释：封装 JobExecutor._flickr_prepare_cmd 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @classmethod
    def _flickr_prepare_cmd(cls, project_root: Path, name: str, root_path: str, payload: dict[str, Any], is_windows: bool) -> list[str]:
        cmd = cls._script_cmd(
            project_root,
            is_windows,
            "prepare_flickr1k.ps1" if name == "flickr1k" else "prepare_flickr30k.ps1",
            "prepare_flickr1k.py" if name == "flickr1k" else "prepare_flickr30k.py",
        )
        if root_path:
            cmd += ["-Root", root_path]
        image_dir = str(payload.get("image_dir", "") or "")
        if image_dir:
            cmd += ["-ImageDir", image_dir]
        if name == "flickr30k":
            captions_source = str(payload.get("captions_source", "") or "")
            if captions_source:
                cmd += ["-CaptionsSource", captions_source]
        else:
            source_root = str(payload.get("captions_source", "") or "")
            if source_root:
                cmd += ["-SourceRoot", source_root]
            output_file = str(payload.get("output_file", "") or "captions_index_single.jsonl")
            if output_file:
                cmd += ["-OutputFile", output_file]
        cmd += ["-AutoDownload", "true" if bool(payload.get("auto_download", True)) else "false"]
        cmd += ["-MaxItems", str(int(payload.get("max_items", 256) or 256))]
        return cmd

    # 中文注释：封装 JobExecutor._coco_prepare_cmd 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @classmethod
    def _coco_prepare_cmd(cls, project_root: Path, root_path: str, payload: dict[str, Any], is_windows: bool) -> list[str]:
        cmd = cls._script_cmd(project_root, is_windows, "prepare_coco_subset.ps1", "prepare_coco_subset.py")
        if root_path:
            cmd += ["-Root", root_path]
        split = str(payload.get("split", "") or "")
        if split:
            cmd += ["-Split", split]
        cmd += ["-MaxItems", str(int(payload.get("max_items", 500) or 500))]
        cmd += ["-AutoDownload", "true" if bool(payload.get("auto_download", True)) else "false"]
        if bool(payload.get("download_annotations", False)):
            cmd += ["-DownloadAnnotations"]
        if bool(payload.get("download_images", False)):
            cmd += ["-DownloadImages"]
        return cmd

    # 中文注释：封装 JobExecutor._register_mini_flickr 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _register_mini_flickr(self, project_root: Path, name: str, root_path: str, log: Callable[[str, str], None]) -> dict[str, Any]:
        dataset_root = project_root / root_path if root_path and not Path(root_path).is_absolute() else Path(root_path or ".")
        image_dir = dataset_root / "images"
        captions_index = dataset_root / "captions_index.jsonl"
        if not image_dir.exists():
            raise RuntimeError(f"dataset prepare failed: {name} missing images")
        if not captions_index.exists():
            raise RuntimeError(f"dataset prepare failed: {name} missing captions_index.jsonl")
        item_count = sum(1 for _ in captions_index.open("r", encoding="utf-8"))
        self.store.upsert_dataset(name=name, root_path=root_path, prepared=True, item_count=item_count, note="prepared via api (demo fixture)")
        log("info", f"registered demo fixture dataset: {dataset_root}")
        return {"dataset": name, "prepared": True, "root_path": root_path, "item_count": item_count}

    # 中文注释：封装 JobExecutor._prepared_item_count 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    @staticmethod
    def _prepared_item_count(project_root: Path, name: str, root_path: str, payload: dict[str, Any]) -> int:
        dataset_root = project_root / root_path if root_path and not Path(root_path).is_absolute() else Path(root_path or ".")
        if name in {"flickr30k", "flickr1k"}:
            default_output = "captions_index_single.jsonl" if name == "flickr1k" else "captions_index.jsonl"
            captions_index = dataset_root / str(payload.get("output_file", "") or default_output)
            return sum(1 for _ in captions_index.open("r", encoding="utf-8")) if captions_index.exists() else 0
        split = str(payload.get("split", "val2017") or "val2017")
        subset_file = dataset_root / "annotations" / f"captions_{split}_subset.json"
        if not subset_file.exists():
            return 0
        try:
            data = json.loads(subset_file.read_text(encoding="utf-8"))
            return len(data.get("annotations", [])) if isinstance(data, dict) else 0
        except (json.JSONDecodeError, OSError):
            return 0

    # 中文注释：封装 JobExecutor._run_dataset_prepare 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _run_dataset_prepare(self, payload: dict[str, Any], log: Callable[[str, str], None]) -> dict[str, Any]:
        name = str(payload.get("name", "")).strip().lower()
        if name not in {"flickr30k", "flickr1k", "coco_subset", "mini_flickr"}:
            raise ValueError("dataset_prepare requires name=flickr30k|flickr1k|coco_subset|mini_flickr")
        project_root = Path(__file__).resolve().parents[3]
        root_path = self._dataset_root_path(name, payload)
        if name == "mini_flickr":
            return self._register_mini_flickr(project_root, name, root_path, log)
        is_windows = platform.system().lower().startswith("win")
        cmd = (
            self._flickr_prepare_cmd(project_root, name, root_path, payload, is_windows)
            if name in {"flickr30k", "flickr1k"}
            else self._coco_prepare_cmd(project_root, root_path, payload, is_windows)
        )
        log("info", f"running dataset prepare script: {' '.join(cmd)}")
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        if p.stdout:
            log("info", p.stdout[-3000:])
        if p.returncode != 0:
            if p.stderr:
                log("error", p.stderr[-3000:])
            raise RuntimeError(f"dataset prepare failed: {name}")

        item_count = self._prepared_item_count(project_root, name, root_path, payload)
        self.store.upsert_dataset(name=name, root_path=root_path, prepared=True, item_count=item_count, note="prepared via api")
        return {"dataset": name, "prepared": True, "root_path": root_path, "item_count": item_count}

    # 中文注释：封装 JobExecutor._execute_generation_job 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _execute_generation_job(
        self,
        *,
        job_type: str,
        config_path: str,
        override: dict[str, Any],
        log: Callable[[str, str], None],
        progress: Callable[[str, str, float | None, str], None],
    ) -> dict[str, Any]:
        task_kind = "vqa" if "vqa" in job_type else "caption"
        return self._run_generation_like(
            config_path=config_path,
            override=override,
            log=log,
            progress=progress,
            force_task_kind=task_kind,
        )

    # 中文注释：封装 JobExecutor._execute_payload_job 的内部步骤，让后端业务服务主流程保持清晰并隔离边界细节。
    def _execute_payload_job(
        self,
        *,
        job_type: str,
        config_path: str,
        payload: dict[str, Any],
        log: Callable[[str, str], None],
    ) -> dict[str, Any] | None:
        if job_type == "run_sweep":
            sweep_path = str(payload.get("sweep_path", "") or "")
            if cmd_run_sweep(config_path, sweep_path) != 0:
                raise RuntimeError("run_sweep failed")
            log("info", "run_sweep success")
            return {"run_id": ""}
        if job_type == "docs_ingest":
            if cmd_ingest_docs(config_path) != 0:
                raise RuntimeError("docs_ingest failed")
            log("info", "docs_ingest success")
            return {"run_id": ""}
        if job_type == "dataset_prepare":
            return self._run_dataset_prepare(payload, log)
        return None

    # 中文注释：实现 JobExecutor.execute 的核心行为，维护后端业务服务在该对象上的调用契约。
    def execute(
        self,
        job: dict[str, Any],
        log: Callable[[str, str], None],
        progress: Callable[[str, str, float | None, str], None],
    ) -> dict[str, Any]:
        job_type = str(job.get("job_type", ""))
        config_path = str(job.get("config_path", "configs/mvp.yaml"))
        override = self._parse_override(str(job.get("override_json", "") or ""))
        payload = self._parse_payload(str(job.get("payload_json", "") or ""))
        benchmark_mode = bool(job.get("benchmark_mode", 0))

        if str((override.get("extra", {}) or {}).get("workflow_type", "")) == "asset_evaluation":
            return run_asset_evaluation(
                store=self.store,
                artifacts_dir=self.artifacts_dir,
                override=override,
                job_id=str(job.get("id", "")),
                log=log,
                progress=progress,
            )

        if job_type == "run_eval":
            return self._run_eval_like(config_path=config_path, override=override, benchmark_mode=False, log=log, progress=progress)

        if job_type == "run_benchmark":
            return self._run_eval_like(config_path=config_path, override=override, benchmark_mode=True, log=log, progress=progress)

        if job_type == "run_vlr":
            return self._run_vlr_like(config_path=config_path, override=override, log=log, progress=progress)

        if job_type in {"run_vqa", "run_caption"}:
            return self._execute_generation_job(
                job_type=job_type,
                config_path=config_path,
                override=override,
                log=log,
                progress=progress,
            )

        if job_type == "train_advclip":
            return self._train_advclip_like(config_path=config_path, override=override, log=log, progress=progress)

        if job_type == "generate_sample_assets":
            return run_sample_generation_only(
                config_path=config_path,
                override=override,
                artifacts_dir=self.artifacts_dir,
                store=self.store,
                log=log,
                progress=progress,
            )

        payload_result = self._execute_payload_job(job_type=job_type, config_path=config_path, payload=payload, log=log)
        if payload_result is not None:
            return payload_result

        raise ValueError(f"unsupported job_type: {job_type}")
