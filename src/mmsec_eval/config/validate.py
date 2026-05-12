# 文件说明：该文件属于配置系统，集中实现 validate 相关逻辑。
from __future__ import annotations

from pathlib import Path

from mmsec_eval.config.schema import AppConfig
from mmsec_eval.exceptions import ConfigError
from mmsec_eval.plugins.registry import list_plugins
from mmsec_eval.runtime import torch_install_hint


SUPPORTED_DATASET_KINDS = {"toy_shapes", "folder_jsonl", "flickr30k", "flickr1k", "coco_subset", "mini_flickr", "generation_jsonl"}


# 中文注释：封装 _validate_plugin 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_plugin(cfg: AppConfig) -> None:
    if cfg.plugins.model_adapter not in list_plugins("model_adapter"):
        raise ConfigError(f"unknown model adapter: {cfg.plugins.model_adapter}")
    if cfg.plugins.attack not in list_plugins("attack"):
        raise ConfigError(f"unknown attack plugin: {cfg.plugins.attack}")
    if cfg.plugins.defense not in list_plugins("defense"):
        raise ConfigError(f"unknown defense plugin: {cfg.plugins.defense}")
    if cfg.plugins.metric not in list_plugins("metric"):
        raise ConfigError(f"unknown metric plugin: {cfg.plugins.metric}")
    if cfg.plugins.judge not in list_plugins("judge"):
        raise ConfigError(f"unknown judge plugin: {cfg.plugins.judge}")


# 中文注释：封装 _validate_dataset 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_dataset(cfg: AppConfig) -> None:
    kind = cfg.dataset.kind
    if kind not in SUPPORTED_DATASET_KINDS:
        raise ConfigError(f"unsupported dataset.kind: {kind}")

    if kind == "generation_jsonl":
        raw_cases_path = str(cfg.task.cases_jsonl or "").strip()
        if not raw_cases_path:
            raise ConfigError("task.cases_jsonl is required when dataset.kind='generation_jsonl'")
        cases_path = Path(raw_cases_path)
        if not cases_path.exists():
            raise ConfigError(f"task.cases_jsonl not found: {cases_path}")
        return

    if kind == "toy_shapes":
        if cfg.dataset.num_samples <= 0:
            raise ConfigError("dataset.num_samples must be > 0")
        if cfg.dataset.image_size < 32:
            raise ConfigError("dataset.image_size must be >= 32")
        return

    if kind == "folder_jsonl":
        if not cfg.dataset.folder_jsonl:
            raise ConfigError("dataset.folder_jsonl is required when dataset.kind='folder_jsonl'")
        p = Path(cfg.dataset.folder_jsonl)
        if not p.exists():
            raise ConfigError(f"dataset.folder_jsonl not found: {p}")
        return

    if kind == "mini_flickr" and not cfg.dataset.root:
        project_root = Path(__file__).resolve().parents[3]
        cfg.dataset.root = str(project_root / "seed" / "data" / "mini_flickr")

    # Benchmark datasets.
    if not cfg.dataset.root:
        raise ConfigError(f"dataset.root is required when dataset.kind='{kind}'")
    if not cfg.dataset.image_dir:
        raise ConfigError(f"dataset.image_dir is required when dataset.kind='{kind}'")
    if not cfg.dataset.captions_file:
        raise ConfigError(f"dataset.captions_file is required when dataset.kind='{kind}'")

    root = Path(cfg.dataset.root)
    if not root.exists():
        raise ConfigError(f"dataset.root not found: {root}")
    image_dir = Path(cfg.dataset.image_dir)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    if not image_dir.exists():
        raise ConfigError(f"dataset.image_dir not found: {image_dir}")
    captions_file = Path(cfg.dataset.captions_file)
    if not captions_file.is_absolute():
        captions_file = root / captions_file
    if not captions_file.exists():
        raise ConfigError(f"dataset.captions_file not found: {captions_file}")


# 中文注释：封装 _generation_case_source 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _generation_case_source(cfg: AppConfig) -> str:
    return f"{cfg.task.cases_jsonl} {cfg.dataset.benchmark_tag}".lower()


# 中文注释：封装 _validate_generation_task_dataset_match 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_generation_task_dataset_match(cfg: AppConfig) -> None:
    if cfg.task.kind not in {"vqa", "caption"}:
        return
    source = _generation_case_source(cfg)
    if cfg.task.kind == "vqa" and "coco_caption_object_val" in source:
        raise ConfigError("VQA task requires a VQA JSONL dataset, not the COCO caption object JSONL")
    if cfg.task.kind == "caption" and ("vqa_v2_coco_val" in source or "coco_object_probe_val" in source):
        raise ConfigError("Caption task requires the COCO caption object JSONL, not a VQA JSONL dataset")


# 中文注释：封装 _validate_task 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_task(cfg: AppConfig) -> None:
    if cfg.task.kind not in {"pairwise", "vlr", "vqa", "caption"}:
        raise ConfigError("task.kind must be 'pairwise', 'vlr', 'vqa', or 'caption'")
    if cfg.task.eval_scope not in {"clean", "image", "text", "joint"}:
        raise ConfigError("task.eval_scope must be one of: clean, image, text, joint")
    if not cfg.task.compare_stages:
        raise ConfigError("task.compare_stages must be non-empty")
    stage_allowed = {"clean", "attacked", "defended", "defended_attack", "defended_clean"}
    for s in cfg.task.compare_stages:
        if str(s) not in stage_allowed:
            raise ConfigError(
                "task.compare_stages entries must be one of: clean, attacked, defended, defended_attack, defended_clean"
            )
    if cfg.task.kind in {"vqa", "caption"} and not str(cfg.task.cases_jsonl or "").strip():
        raise ConfigError("task.cases_jsonl is required when task.kind is 'vqa' or 'caption'")
    _validate_generation_task_dataset_match(cfg)
    if cfg.task.kind == "vlr" and not cfg.task.retrieval_k:
        raise ConfigError("task.retrieval_k must be non-empty when task.kind='vlr'")
    for k in cfg.task.retrieval_k:
        if int(k) <= 0:
            raise ConfigError("task.retrieval_k entries must be > 0")


# 中文注释：封装 _validate_attack 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_attack(cfg: AppConfig) -> None:
    if cfg.attack.mode not in {"A", "B"}:
        raise ConfigError("attack.mode must be 'A' or 'B'")
    positive_fields = {
        "steps": cfg.attack.steps,
        "patch_size": cfg.attack.patch_size,
        "topk": cfg.attack.topk,
        "epochs": cfg.attack.epochs,
        "batch_size": cfg.attack.batch_size,
        "text_candidates_k": cfg.attack.text_candidates_k,
        "topology_k": cfg.attack.topology_k,
        "gan_steps": cfg.attack.gan_steps,
        "kernel_size": cfg.attack.kernel_size,
        "kernel_sigma": cfg.attack.kernel_sigma,
        "severity": cfg.attack.severity,
        "timeout_sec": cfg.attack.timeout_sec,
    }
    for field, value in positive_fields.items():
        if value <= 0:
            raise ConfigError(f"attack.{field} must be > 0")

    non_negative_fields = {
        "eps_t": cfg.attack.eps_t,
        "patch_train_steps": cfg.attack.patch_train_steps,
        "momentum_decay": cfg.attack.momentum_decay,
        "variance_samples": cfg.attack.variance_samples,
        "variance_radius": cfg.attack.variance_radius,
        "nesterov_scale": cfg.attack.nesterov_scale,
        "cw_const": cfg.attack.cw_const,
        "cw_confidence": cfg.attack.cw_confidence,
        "crop_scale": cfg.attack.crop_scale,
        "crop_ratio": cfg.attack.crop_ratio,
    }
    for field, value in non_negative_fields.items():
        if value < 0:
            raise ConfigError(f"attack.{field} must be >= 0")

    if cfg.attack.diversity_prob < 0 or cfg.attack.diversity_prob > 1:
        raise ConfigError("attack.diversity_prob must be in [0, 1]")
    if cfg.attack.resize_rate <= 0 or cfg.attack.resize_rate > 1:
        raise ConfigError("attack.resize_rate must be in (0, 1]")
    if cfg.attack.severity < 1 or cfg.attack.severity > 5:
        raise ConfigError("attack.severity must be in [1, 5]")
    if cfg.attack.crop_scale <= 0:
        raise ConfigError("attack.crop_scale must be > 0")
    if cfg.attack.crop_ratio <= 0:
        raise ConfigError("attack.crop_ratio must be > 0")
    for k, v in cfg.attack.loss_weights.items():
        if v < 0:
            raise ConfigError(f"attack.loss_weights.{k} must be >= 0")


# 中文注释：封装 _validate_defense 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_defense(cfg: AppConfig) -> None:
    bounded_fields = {
        "jpeg_quality": (cfg.defense.jpeg_quality, 1, 100),
        "bit_depth": (cfg.defense.bit_depth, 1, 8),
        "strong_resize_ratio": (cfg.defense.strong_resize_ratio, 0, 1),
        "strong_jpeg_quality": (cfg.defense.strong_jpeg_quality, 1, 100),
        "strong_bit_depth": (cfg.defense.strong_bit_depth, 1, 8),
    }
    if cfg.defense.resize_ratio <= 0 or cfg.defense.resize_ratio > 1:
        raise ConfigError("defense.resize_ratio must be in (0, 1]")
    for field, (value, lower, upper) in bounded_fields.items():
        if field == "strong_resize_ratio":
            if value <= lower or value > upper:
                raise ConfigError("defense.strong_resize_ratio must be in (0, 1]")
        elif value < lower or value > upper:
            raise ConfigError(f"defense.{field} must be in [{lower}, {upper}]")

    non_negative_fields = {
        "blur_sigma": cfg.defense.blur_sigma,
        "median_kernel": cfg.defense.median_kernel,
        "strong_blur_sigma": cfg.defense.strong_blur_sigma,
        "selection_penalty": cfg.defense.selection_penalty,
        "text_repair_max_edits": cfg.defense.text_repair_max_edits,
    }
    for field, value in non_negative_fields.items():
        if value < 0:
            raise ConfigError(f"defense.{field} must be >= 0")
    if cfg.defense.median_kernel > 0 and cfg.defense.median_kernel % 2 == 0:
        raise ConfigError("defense.median_kernel must be odd when enabled")
    if cfg.defense.text_candidates_k <= 0:
        raise ConfigError("defense.text_candidates_k must be > 0")


# 中文注释：封装 _validate_runtime 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_runtime(cfg: AppConfig) -> None:
    dev = str(cfg.runtime.device or "").strip().lower()
    if not dev:
        raise ConfigError("runtime.device must be set to 'cuda' or 'cuda:0' (GPU-only).")
    if dev == "auto":
        raise ConfigError("runtime.device='auto' is not allowed. Use 'cuda' or 'cuda:0' (GPU-only).")
    if dev == "cpu":
        raise ConfigError("runtime.device='cpu' is not allowed. Use 'cuda' or 'cuda:0' (GPU-only).")
    if not dev.startswith("cuda"):
        raise ConfigError("runtime.device must start with 'cuda' (e.g. cuda, cuda:0).")
    if bool(cfg.runtime.fallback_cpu):
        raise ConfigError("runtime.fallback_cpu must be false (CPU fallback is disabled).")

    try:
        import torch
    except (ImportError, OSError) as e:  # pragma: no cover
        raise ConfigError(f"CUDA is required but torch is not installed/usable: {e}") from e

    if getattr(torch.version, "cuda", None) is None or not torch.cuda.is_available():
        raise ConfigError(
            "CUDA is required but torch CUDA is unavailable. "
            f"torch={getattr(torch, '__version__', 'unknown')} torch.version.cuda={getattr(torch.version, 'cuda', None)}. "
            f"Fix: {torch_install_hint()}"
        )
    if cfg.runtime.num_workers < 0:
        raise ConfigError("runtime.num_workers must be >= 0")


# 中文注释：封装 _validate_model_and_plugins 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_model_and_plugins(cfg: AppConfig) -> None:
    if cfg.plugins.model_adapter == "http" and not cfg.model.http_endpoint:
        raise ConfigError("model.http_endpoint is required when model_adapter=http")
    if cfg.plugins.model_adapter == "openai_compat" and not cfg.model.openai_model_name:
        raise ConfigError("model.openai_model_name is required when model_adapter=openai_compat")
    if cfg.plugins.model_adapter == "gemini_vision" and not cfg.model.gemini_model_name:
        raise ConfigError("model.gemini_model_name is required when model_adapter=gemini_vision")
    _validate_plugin(cfg)


# 中文注释：封装 _validate_runner 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_runner(cfg: AppConfig) -> None:
    adapters = set(list_plugins("model_adapter"))
    if cfg.runner.surrogate_model_adapter and cfg.runner.surrogate_model_adapter not in adapters:
        raise ConfigError(f"unknown runner.surrogate_model_adapter: {cfg.runner.surrogate_model_adapter}")
    for name in cfg.runner.victim_model_adapters:
        if name not in adapters:
            raise ConfigError(f"unknown runner.victim_model_adapters entry: {name}")
    if cfg.runner.max_pairs < 0:
        raise ConfigError("runner.max_pairs must be >= 0")
    if cfg.runner.experiment_id and len(str(cfg.runner.experiment_id)) > 128:
        raise ConfigError("runner.experiment_id is too long")
    if bool(cfg.runner.continue_on_error):
        raise ConfigError("runner.continue_on_error must be false in strict real mode.")


# 中文注释：封装 _validate_risk 的内部步骤，让配置系统主流程保持清晰并隔离边界细节。
def _validate_risk(cfg: AppConfig) -> None:
    if cfg.risk.l2_reference <= 0:
        raise ConfigError("risk.l2_reference must be > 0")
    if cfg.risk.linf_reference <= 0:
        raise ConfigError("risk.linf_reference must be > 0")
    if cfg.risk.rank_reference <= 0:
        raise ConfigError("risk.rank_reference must be > 0")
    threshold = float(cfg.risk.transfer_success_threshold)
    if threshold < 0 or threshold > 1:
        raise ConfigError("risk.transfer_success_threshold must be between 0 and 1")
    for k, v in cfg.risk.weights.items():
        if float(v) < 0:
            raise ConfigError(f"risk.weights.{k} must be >= 0")


# 中文注释：实现 validate_config 的核心流程，支撑配置系统中的业务语义和异常边界。
def validate_config(cfg: AppConfig) -> None:
    if cfg.seed < 0:
        raise ConfigError("seed must be >= 0")
    _validate_task(cfg)
    _validate_attack(cfg)
    _validate_defense(cfg)
    _validate_dataset(cfg)
    _validate_runtime(cfg)
    if cfg.bootstrap.max_warmup_minutes <= 0:
        raise ConfigError("bootstrap.max_warmup_minutes must be > 0")
    _validate_model_and_plugins(cfg)
    _validate_runner(cfg)
    _validate_risk(cfg)
