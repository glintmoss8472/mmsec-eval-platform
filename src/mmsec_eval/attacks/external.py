from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mmsec_eval.attacks.base import clip01, perturb_stats
from mmsec_eval.plugins.base import AttackPlugin
from mmsec_eval.types import AttackContext, AttackedSample, AttackTraceStep, Sample


class ExternalAttackPrerequisiteError(RuntimeError):
    pass


class ExternalAttackExecutionError(RuntimeError):
    pass


TRACEABLE_EXTERNAL_ERRORS = (
    ExternalAttackPrerequisiteError,
    ExternalAttackExecutionError,
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    ImportError,
    KeyError,
    subprocess.SubprocessError,
)


@dataclass(frozen=True)
class ExternalCommandSpec:
    name: str
    attack_mode: str
    checkpoint_field: str = ""
    checkpoint_label: str = "checkpoint_path"
    requires_checkpoint: bool = False
    requires_target: bool = False
    target_description: str = "target_image or target_text"
    supports_disable_wandb: bool = False
    requires_target_image: bool = False


def _cfg(ctx: AttackContext, attack_name: str, key: str, default: Any = None) -> Any:
    extra = getattr(ctx.config, "extra", {}) or {}
    if isinstance(extra, dict):
        external = extra.get("external_attacks", {}) or {}
        if isinstance(external, dict) and isinstance(external.get(attack_name), dict) and key in external[attack_name]:
            return external[attack_name][key]
        if isinstance(extra.get(attack_name), dict) and key in extra[attack_name]:
            return extra[attack_name][key]
    value = getattr(getattr(ctx.config, "attack", object()), key, default)
    return default if value is None else value


def _str_cfg(ctx: AttackContext, attack_name: str, key: str, default: str = "") -> str:
    return str(_cfg(ctx, attack_name, key, default) or "").strip()


def _to_pil(image: np.ndarray) -> Image.Image:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float32), 0, 1) * 255).round().astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"expected image HWC array, got {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return Image.fromarray(arr[:, :, :3], mode="RGB")


def _from_pil(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(image).save(path)


def _load(path: Path) -> np.ndarray:
    return _from_pil(Image.open(path))


def _resize_like_clean(adv: np.ndarray, clean: np.ndarray, trace: dict[str, Any] | None = None) -> np.ndarray:
    adv = clip01(np.asarray(adv, dtype=np.float32))
    clean_arr = np.asarray(clean, dtype=np.float32)
    if adv.shape[:2] == clean_arr.shape[:2]:
        return adv
    before = list(adv.shape)
    resized = _from_pil(_to_pil(adv).resize((int(clean_arr.shape[1]), int(clean_arr.shape[0])), _resample("BILINEAR")))
    if trace is not None:
        trace["output_shape_before_resize"] = before
        trace["resized_to_input_shape"] = list(resized.shape)
    return resized


def _tail(text: str, chars: int = 4000) -> str:
    text = str(text or "")
    return text if len(text) <= chars else text[-chars:]


def _safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value


def _write_trace(path: Path, trace: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(trace), ensure_ascii=False, indent=2), encoding="utf-8")


def _debug_dir(ctx: AttackContext, attack_name: str, sample: Sample) -> Path:
    raw = str(getattr(ctx, "sample_debug_dir", "") or "").strip()
    if raw:
        path = Path(raw)
    else:
        root = Path(str(getattr(ctx, "run_dir", "") or getattr(ctx.config, "artifacts_dir", "artifacts") or "artifacts"))
        sample_id = str(sample.sample_id or "sample").replace("/", "_").replace("\\", "_")
        path = root / "attack_debug" / sample_id
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _attacked(clean_sample: Sample, adv: np.ndarray, attack_name: str, attack_mode: str, trace: dict[str, Any], refs: dict[str, str]) -> AttackedSample:
    adv = _resize_like_clean(adv, np.asarray(clean_sample.image, dtype=np.float32), trace)
    l0, l2, linf = perturb_stats(np.asarray(clean_sample.image, dtype=np.float32), adv)
    sample = Sample(clean_sample.sample_id, adv, clean_sample.text, clean_sample.target_text, {**clean_sample.metadata, "attack": attack_name})
    metadata = {
        "attack_name": attack_name,
        "attack_mode": attack_mode,
        "attack_scope": "image",
        "perturbation_l0": l0,
        "perturbation_l2": l2,
        "perturbation_linf": linf,
        "attack_trace": [_safe_json(trace)],
    }
    return AttackedSample(
        sample=sample,
        perturbation_l2=l2,
        perturbation_linf=linf,
        perturbation_l0=l0,
        attack_trace=[AttackTraceStep(step=0, loss_total=0.0, metadata=_safe_json(trace))],
        artifact_refs=refs,
        metadata=metadata,
    )


def _existing(raw: str, label: str, required: bool) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        if required:
            raise ExternalAttackPrerequisiteError(f"{label} is required")
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        if required:
            raise ExternalAttackPrerequisiteError(f"{label} not found: {path}")
        return None
    return path


def _resample(name: str) -> int:
    return int(getattr(getattr(Image, "Resampling", Image), name))


class XTransferUAPAttack(AttackPlugin):
    name = "xtransfer_uap"
    attack_mode = "universal_perturbation"

    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        start = time.monotonic()
        debug = _debug_dir(ctx, self.name, sample)
        trace_path = debug / f"{self.name}_trace.json"
        uap_path = _str_cfg(ctx, self.name, "uap_path", "")
        repo_dir = _str_cfg(ctx, self.name, "repo_dir", "")
        uap_name = _str_cfg(ctx, self.name, "uap_name", "xtransfer_large_linf_eps12_non_targeted")
        epsilon = float(_cfg(ctx, self.name, "epsilon", 12 / 255) or 12 / 255)
        threat_model = self._normalize_threat_model(_str_cfg(ctx, self.name, "threat_model", "linf_non_targeted") or "linf_non_targeted")
        cache_dir = _str_cfg(ctx, self.name, "cache_dir", "")
        device = _str_cfg(ctx, self.name, "device", getattr(getattr(ctx.config, "runtime", object()), "device", "cuda")) or "cuda"
        trace = {
            "attack_name": self.name,
            "attack_mode": self.attack_mode,
            "attack_scope": "image",
            "uap_name": uap_name,
            "threat_model": threat_model,
            "epsilon": epsilon,
            "repo_dir": repo_dir,
            "cache_dir": cache_dir,
            "device": device,
            "used_local_uap_path": bool(uap_path),
            "used_package_zoo": False,
            "used_official_zoo": False,
            "status": "starting",
        }
        try:
            repo_path = self._prepare_repo(repo_dir) if repo_dir else None
            if uap_path:
                delta = self._load_local(_existing(uap_path, "xtransfer_uap uap_path", True))
                adv = self._apply(np.asarray(sample.image, dtype=np.float32), delta, epsilon, threat_model)
            else:
                adv = self._apply_official_zoo(np.asarray(sample.image, dtype=np.float32), uap_name, threat_model, epsilon, cache_dir, device)
                trace["used_package_zoo"] = True
                trace["used_official_zoo"] = True
            if repo_path:
                trace["official_repo_dir"] = str(repo_path)
            out = debug / f"{self.name}.png"
            _save(out, adv)
            trace.update({"status": "success", "elapsed_sec": time.monotonic() - start, "output_image": str(out)})
            _write_trace(trace_path, trace)
            return _attacked(sample, adv, self.name, self.attack_mode, trace, {"adv_image": str(out), "trace": str(trace_path)})
        except TRACEABLE_EXTERNAL_ERRORS as exc:
            trace.update({
                "status": "failed_precondition" if isinstance(exc, ExternalAttackPrerequisiteError) else "failed",
                "elapsed_sec": time.monotonic() - start,
                "failure_reason": str(exc),
            })
            _write_trace(trace_path, trace)
            raise

    def _prepare_repo(self, repo_dir: str) -> Path:
        repo = Path(repo_dir).expanduser()
        if not repo.exists():
            raise ExternalAttackPrerequisiteError(f"xtransfer_uap repo_dir not found: {repo}")
        for candidate in (repo / "src", repo):
            text = str(candidate.resolve())
            if candidate.exists() and text not in sys.path:
                sys.path.insert(0, text)
        return repo

    def _normalize_threat_model(self, value: str) -> str:
        normalized = str(value or "").strip().lower().replace("-", "_")
        mapping = {
            "linf": "linf_non_targeted",
            "l_inf": "linf_non_targeted",
            "infinity": "linf_non_targeted",
            "l2": "l2_non_targeted",
            "advclip": "unrestricted_non_targeted",
            "unrestricted": "unrestricted_non_targeted",
        }
        return mapping.get(normalized, normalized or "linf_non_targeted")

    def _load_local(self, path: Path | None) -> np.ndarray:
        assert path is not None
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path), dtype=np.float32)
        if path.suffix.lower() == ".npz":
            z = np.load(path)
            key = next((k for k in ("uap", "perturbation", "delta", "noise") if k in z), list(z.keys())[0])
            return np.asarray(z[key], dtype=np.float32)
        if path.suffix.lower() in {".pt", ".pth"}:
            import torch
            value = torch.load(path, map_location="cpu")
            if isinstance(value, dict):
                value = next((value[k] for k in ("uap", "perturbation", "delta", "noise") if k in value), next(iter(value.values())))
            return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=np.float32)
        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file
            value = load_file(str(path))
            value = next((value[k] for k in ("uap", "perturbation", "delta", "noise") if k in value), next(iter(value.values())))
            return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=np.float32)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            return _load(path)
        raise ExternalAttackPrerequisiteError(f"unsupported xtransfer_uap uap_path suffix: {path.suffix}")

    def _apply_official_zoo(self, clean: np.ndarray, uap_name: str, threat_model: str, epsilon: float, cache_dir: str, device: str) -> np.ndarray:
        if cache_dir:
            cache = str(Path(cache_dir).expanduser())
            os.environ.setdefault("HF_HOME", cache)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache) / "hub"))
        os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
        try:
            import torch
        except ImportError as exc:
            raise ExternalAttackPrerequisiteError("PyTorch is required for XTransferBench official zoo execution") from exc
        try:
            zoo = importlib.import_module("XTransferBench.zoo")
        except ImportError as exc:
            raise ExternalAttackPrerequisiteError(
                "XTransferBench is not importable. Install it with `pip install XTransferBench`, `pip install -e external/XTransferBench`, or set attack.repo_dir to the cloned repository."
            ) from exc
        try:
            attacker = zoo.load_attacker(threat_model, uap_name)
        except KeyError as exc:
            available = []
            try:
                available = list(zoo.list_attacker(threat_model))
            except (AttributeError, ImportError, KeyError, RuntimeError, ValueError) as list_exc:
                available = []
                _ = list_exc
            raise ExternalAttackExecutionError(f"XTransferBench UAP {uap_name!r} is not available for threat_model={threat_model!r}; available={available[:20]}") from exc
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ExternalAttackExecutionError(f"failed to load XTransferBench official UAP {uap_name!r}: {exc}") from exc
        if hasattr(attacker, "interpolate_epsilon"):
            try:
                attacker.interpolate_epsilon(float(epsilon))
            except (AttributeError, RuntimeError, TypeError, ValueError) as interp_exc:
                _ = interp_exc
        tensor = torch.from_numpy(clean.transpose(2, 0, 1)).unsqueeze(0).float()
        dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        attacker = attacker.eval().to(dev)
        with torch.no_grad():
            out = attacker(tensor.to(dev))
        arr = out.detach().cpu().clamp(0, 1).squeeze(0).numpy().transpose(1, 2, 0)
        return clip01(arr)

    def _apply(self, clean: np.ndarray, delta: np.ndarray, epsilon: float, threat_model: str) -> np.ndarray:
        delta = np.squeeze(np.asarray(delta, dtype=np.float32))
        if delta.ndim == 2:
            delta = np.repeat(delta[:, :, None], clean.shape[2], axis=2)
        if delta.ndim == 3 and delta.shape[0] in {1, 3} and delta.shape[-1] not in {1, 3}:
            delta = np.transpose(delta, (1, 2, 0))
        if delta.ndim != 3:
            raise ExternalAttackExecutionError(f"UAP tensor must be image-like, got shape={delta.shape}")
        if delta.shape[2] == 1:
            delta = np.repeat(delta, clean.shape[2], axis=2)
        delta = delta[:, :, : clean.shape[2]]
        if delta.shape[:2] != clean.shape[:2]:
            chans = [
                np.asarray(Image.fromarray(delta[:, :, i].astype(np.float32), mode="F").resize((clean.shape[1], clean.shape[0]), _resample("BILINEAR")), dtype=np.float32)
                for i in range(delta.shape[2])
            ]
            delta = np.stack(chans, axis=2)
        if float(np.nanmax(np.abs(delta))) > 1.0:
            delta = delta / 255.0
        if threat_model.lower() in {"linf", "l_inf", "l_infinity", "linf_non_targeted", "linf_targeted", "infinity"}:
            delta = np.clip(delta, -epsilon, epsilon)
        return clip01(clean + delta)


class ExternalCommandAttack(AttackPlugin):
    spec: ExternalCommandSpec

    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        start = time.monotonic(); debug = _debug_dir(ctx, self.spec.name, sample); trace_path = debug / f"{self.spec.name}_trace.json"
        trace: dict[str, Any] = {"attack_name": self.spec.name, "attack_mode": self.spec.attack_mode, "attack_scope": "image", "status": "starting"}
        try:
            repo = _existing(_str_cfg(ctx, self.spec.name, "repo_dir", ""), f"{self.spec.name} repo_dir", True)
            assert repo is not None
            output_dir = Path(_str_cfg(ctx, self.spec.name, "output_dir", "") or debug).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
            input_path = (debug / "clean_input.png").resolve(); output_path = (output_dir / f"{self.spec.name}_{sample.sample_id or 'sample'}.png").resolve(); _save(input_path, sample.image)
            checkpoint = None
            if self.spec.checkpoint_field:
                raw = _str_cfg(ctx, self.spec.name, self.spec.checkpoint_field, "") or _str_cfg(ctx, self.spec.name, "checkpoint_path", "")
                checkpoint = _existing(raw, f"{self.spec.name} {self.spec.checkpoint_label}", self.spec.requires_checkpoint)
            target_image_raw = _str_cfg(ctx, self.spec.name, "target_image", "")
            target_image = _existing(target_image_raw, f"{self.spec.name} target_image", False) if target_image_raw else None
            target_text = _str_cfg(ctx, self.spec.name, "target_text", "") or str(sample.target_text or "").strip()
            if self.spec.requires_target_image and target_image is None:
                raise ExternalAttackPrerequisiteError(f"{self.spec.name} requires target_image")
            if self.spec.requires_target and target_image is None and not target_text:
                raise ExternalAttackPrerequisiteError(f"{self.spec.name} requires {self.spec.target_description}")
            template = _str_cfg(ctx, self.spec.name, "command_template", "")
            if not template:
                raise ExternalAttackPrerequisiteError(f"{self.spec.name} command_template is required")
            timeout = float(_cfg(ctx, self.spec.name, "timeout_sec", 1800) or 1800)
            command = self._command(template, ctx, repo, input_path, output_path, checkpoint, target_image, target_text, sample)
            trace.update({"repo_dir": str(repo), "command": command, "timeout_sec": timeout, "target_image": str(target_image) if target_image else "", "target_text": target_text, "output_image": str(output_path), "device": _str_cfg(ctx, self.spec.name, "device", getattr(getattr(ctx.config, "runtime", object()), "device", "cuda")), **self._extra_trace(ctx)})
            if checkpoint:
                trace[self.spec.checkpoint_label] = checkpoint.name; trace[f"{self.spec.checkpoint_label}_path"] = str(checkpoint)
            env = os.environ.copy()
            hf_endpoint = _str_cfg(ctx, self.spec.name, "hf_endpoint", "https://hf-mirror.com")
            if hf_endpoint:
                env.setdefault("HF_ENDPOINT", hf_endpoint)
            cache_dir = _str_cfg(ctx, self.spec.name, "cache_dir", "")
            if cache_dir:
                cache_path = str(Path(cache_dir).expanduser().resolve())
                env.setdefault("HF_HOME", cache_path)
                env.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache_path) / "hub"))
                env.setdefault("TRANSFORMERS_CACHE", str(Path(cache_path) / "hub"))
            env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
            env.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            if self.spec.supports_disable_wandb and bool(_cfg(ctx, self.spec.name, "disable_wandb", True)):
                env["WANDB_MODE"] = "disabled"; env["WANDB_DISABLED"] = "true"; trace["disable_wandb"] = True
            proc = subprocess.run(command, shell=True, cwd=str(repo), timeout=timeout, capture_output=True, text=True, env=env)
            trace.update({"returncode": proc.returncode, "stdout_tail": _tail(proc.stdout), "stderr_tail": _tail(proc.stderr), "elapsed_sec": time.monotonic() - start})
            if proc.returncode != 0:
                trace.update({"status": "failed", "failure_reason": f"external command returned non-zero exit code {proc.returncode}"}); _write_trace(trace_path, trace)
                raise ExternalAttackExecutionError(f"{self.spec.name} failed: returncode={proc.returncode}; stderr_tail={_tail(proc.stderr, 1000)}")
            if not output_path.exists():
                trace.update({"status": "missing_output", "failure_reason": f"expected output image was not created: {output_path}"}); _write_trace(trace_path, trace)
                raise ExternalAttackExecutionError(f"{self.spec.name} missing output image: {output_path}")
            external_trace_path = output_path.with_name(f"{output_path.stem}_external_trace.json")
            if external_trace_path.exists():
                try:
                    external_trace = json.loads(external_trace_path.read_text(encoding="utf-8"))
                    trace["external_trace_json"] = str(external_trace_path)
                    trace["external_trace"] = external_trace
                    for key in ("source", "official_module", "requested_corruption_type", "official_transformation", "severity", "seed", "actual_params"):
                        if key in external_trace:
                            trace[key] = external_trace[key]
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    trace["external_trace_read_error"] = str(exc)
            adv = _resize_like_clean(_load(output_path), np.asarray(sample.image, dtype=np.float32), trace)
            if trace.get("resized_to_input_shape"):
                _save(output_path, adv)
            trace["status"] = "success"; _write_trace(trace_path, trace)
            return _attacked(sample, adv, self.spec.name, self.spec.attack_mode, trace, {"adv_image": str(output_path), "trace": str(trace_path)})
        except TRACEABLE_EXTERNAL_ERRORS as exc:
            trace.setdefault("elapsed_sec", time.monotonic() - start)
            trace.setdefault("status", "failed_precondition" if isinstance(exc, ExternalAttackPrerequisiteError) else "failed")
            trace.setdefault("failure_reason", str(exc)); _write_trace(trace_path, trace); raise

    def _python(self, ctx: AttackContext) -> str:
        conda_env = _str_cfg(ctx, self.spec.name, "conda_env", "")
        python_bin = _str_cfg(ctx, self.spec.name, "python_bin", "") or sys.executable or "python"
        return f"conda run -n {shlex.quote(conda_env)} python" if conda_env else shlex.quote(python_bin)

    def _command(self, template: str, ctx: AttackContext, repo: Path, input_path: Path, output_path: Path, checkpoint: Path | None, target_image: Path | None, target_text: str, sample: Sample) -> str:
        values: dict[str, Any] = {
            "python": self._python(ctx), "project_root": shlex.quote(str(Path(__file__).resolve().parents[3])), "repo_dir": shlex.quote(str(repo)), "input_image": shlex.quote(str(input_path)), "clean_image": shlex.quote(str(input_path)), "output_image": shlex.quote(str(output_path)), "output_dir": shlex.quote(str(output_path.parent)), "external_trace": shlex.quote(str(output_path.with_name(f"{output_path.stem}_external_trace.json"))), "target_image": shlex.quote(str(target_image)) if target_image else "", "target_text": shlex.quote(target_text), "source_text": shlex.quote(str(sample.text or "")), "sample_text": shlex.quote(str(sample.text or "")), "device": shlex.quote(_str_cfg(ctx, self.spec.name, "device", getattr(getattr(ctx.config, "runtime", object()), "device", "cuda")) or "cuda"), "steps": str(int(_cfg(ctx, self.spec.name, "steps", 0) or 0)), "epsilon": str(float(_cfg(ctx, self.spec.name, "epsilon", 0) or 0)), "alpha": str((lambda v: v * 255.0 if v < 1.0 else v)(float(_cfg(ctx, self.spec.name, "step_size", _cfg(ctx, self.spec.name, "alpha", 1.0)) or 1.0))), "corruption_type": shlex.quote(_str_cfg(ctx, self.spec.name, "corruption_type", "gaussian_noise") or "gaussian_noise"), "severity": str(int(_cfg(ctx, self.spec.name, "severity", 2) or 2)), "corruption_seed": str(int(_cfg(ctx, self.spec.name, "corruption_seed", _cfg(ctx, self.spec.name, "seed", getattr(ctx.config, "seed", 42))) or 42)), "seed": str(int(_cfg(ctx, self.spec.name, "corruption_seed", _cfg(ctx, self.spec.name, "seed", getattr(ctx.config, "seed", 42))) or 42)), "surrogate_models": shlex.quote(",".join(str(x) for x in (_cfg(ctx, self.spec.name, "surrogate_models", []) or []))), "ensemble_models": shlex.quote(",".join(str(x) for x in (_cfg(ctx, self.spec.name, "ensemble_models", []) or []))), "clip_backbones": shlex.quote(",".join(str(x) for x in (_cfg(ctx, self.spec.name, "clip_backbones", []) or []))), "crop_scale": str(float(_cfg(ctx, self.spec.name, "crop_scale", 1) or 1)), "crop_ratio": str(float(_cfg(ctx, self.spec.name, "crop_ratio", 1) or 1)), "input_res": str(int(_cfg(ctx, self.spec.name, "input_res", 224) or 224)), "lam": str(float(_cfg(ctx, self.spec.name, "lam", 0.6) or 0.6)), "tau": str(float(_cfg(ctx, self.spec.name, "tau", 0.2) or 0.2)), "omega": str(float(_cfg(ctx, self.spec.name, "omega", 2.0) or 2.0)), "input_format": shlex.quote(_str_cfg(ctx, self.spec.name, "input_format", "image") or "image"),
        }
        if checkpoint:
            values["checkpoint_path"] = shlex.quote(str(checkpoint)); values[self.spec.checkpoint_field or "checkpoint_path"] = shlex.quote(str(checkpoint))
        try:
            return template.format(**values)
        except KeyError as exc:
            raise ExternalAttackPrerequisiteError(f"{self.spec.name} command_template references unknown placeholder: {exc}") from exc

    def _extra_trace(self, ctx: AttackContext) -> dict[str, Any]:
        return {}


class VQAVisualCorruptionAttack(ExternalCommandAttack):
    spec = ExternalCommandSpec("vqa_visual_corruption", "official_visual_corruption")

    def _extra_trace(self, ctx: AttackContext) -> dict[str, Any]:
        severity = int(_cfg(ctx, self.spec.name, "severity", 2) or 2)
        if not 1 <= severity <= 5:
            raise ValueError("vqa_visual_corruption severity must be in [1, 5]")
        return {
            "official_framework": "VQA Visual Robustness Benchmark",
            "official_repo_url": "https://github.com/ishmamt/VQA-Visual-Robustness-Benchmark",
            "corruption_type": _str_cfg(ctx, self.spec.name, "corruption_type", "gaussian_noise") or "gaussian_noise",
            "severity": severity,
            "seed": int(_cfg(ctx, self.spec.name, "corruption_seed", _cfg(ctx, self.spec.name, "seed", getattr(ctx.config, "seed", 42))) or 42),
            "training": False,
            "requires_checkpoint": False,
            "full_dataset_generation": False,
        }


class FOAAttack(ExternalCommandAttack):
    spec = ExternalCommandSpec("foa_attack", "targeted_transfer", requires_target=True)
    def _extra_trace(self, ctx: AttackContext) -> dict[str, Any]:
        return {"surrogate_models": _cfg(ctx, self.spec.name, "surrogate_models", []) or [], "steps": int(_cfg(ctx, self.spec.name, "steps", 0) or 0), "epsilon": float(_cfg(ctx, self.spec.name, "epsilon", 0) or 0), "blackbox_evaluation": False}


class AnyAttackPlugin(ExternalCommandAttack):
    spec = ExternalCommandSpec("anyattack", "pretrained_generator_targeted", "decoder_path", "decoder_path", True, True)
    def _extra_trace(self, ctx: AttackContext) -> dict[str, Any]:
        return {"training": False, "decoder_path_basename": Path(_str_cfg(ctx, self.spec.name, "decoder_path", "")).name}


class MPCAttackPlugin(ExternalCommandAttack):
    spec = ExternalCommandSpec("mpc_attack", "multi_paradigm_collaborative_transfer", requires_target=True, target_description="target_image", requires_target_image=True)

    def _extra_trace(self, ctx: AttackContext) -> dict[str, Any]:
        return {
            "surrogate_models": _cfg(ctx, self.spec.name, "surrogate_models", []) or [],
            "clip_backbones": _cfg(ctx, self.spec.name, "clip_backbones", []) or [],
            "steps": int(_cfg(ctx, self.spec.name, "steps", 0) or 0),
            "epsilon": float(_cfg(ctx, self.spec.name, "epsilon", 0) or 0),
            "lam": float(_cfg(ctx, self.spec.name, "lam", 0.6) or 0.6),
            "tau": float(_cfg(ctx, self.spec.name, "tau", 0.2) or 0.2),
            "omega": float(_cfg(ctx, self.spec.name, "omega", 2.0) or 2.0),
            "local_matching": True,
            "multi_paradigm": True,
            "training": False,
            "blackbox_evaluation": False,
        }


class MAttackPlugin(ExternalCommandAttack):
    spec = ExternalCommandSpec("m_attack", "targeted_transfer_local_semantic", requires_target=True, supports_disable_wandb=True)
    def _extra_trace(self, ctx: AttackContext) -> dict[str, Any]:
        return {"ensemble_models": _cfg(ctx, self.spec.name, "ensemble_models", []) or [], "steps": int(_cfg(ctx, self.spec.name, "steps", 0) or 0), "epsilon": float(_cfg(ctx, self.spec.name, "epsilon", 0) or 0), "crop_scale": float(_cfg(ctx, self.spec.name, "crop_scale", 1) or 1), "crop_ratio": float(_cfg(ctx, self.spec.name, "crop_ratio", 1) or 1), "local_matching": True, "model_ensemble": True, "blackbox_evaluation": False}
