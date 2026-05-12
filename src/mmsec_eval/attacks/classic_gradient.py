from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmsec_eval.attacks.base import perturb_stats
from mmsec_eval.plugins.base import AttackPlugin
from mmsec_eval.types import AttackContext, AttackTraceStep, AttackedSample, Sample

try:
    import torchattacks  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torchattacks = None


@dataclass(frozen=True)
class ClassicAttackSpec:
    key: str
    steps_override: int | None = None
    random_start: bool = False
    use_momentum: bool = False
    nesterov: bool = False
    input_diversity: bool = False
    translation_invariant: bool = False
    variance_tuning: bool = False
    cw_style: bool = False
    torchattack_name: str | None = None


def _seed(sample_id: str, attack_name: str, mode: str) -> int:
    raw = f"classic:{attack_name}:{sample_id}:{mode}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest(), 16) % (2**31 - 1)


def _require_gradient_adapter(adapter: Any) -> str:
    if not hasattr(adapter, "score_pairs_torch"):
        raise RuntimeError("classic gradient attacks require adapter.score_pairs_torch")
    device = str(getattr(adapter, "_device", "") or "")
    if not device:
        raise RuntimeError("classic gradient attacks require a torch-backed surrogate adapter")
    return device


def _to_tensor(image: np.ndarray, device: str):
    import torch

    return torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)


def _score_loss(adapter: Any, image_bchw, text: str | list[str]):
    batch_size = int(image_bchw.shape[0])
    if isinstance(text, (list, tuple)):
        texts = [str(item) for item in text]
        if len(texts) != batch_size:
            raise ValueError(f"text batch length mismatch: texts={len(texts)} images={batch_size}")
    else:
        texts = [str(text)] * batch_size
    scores = adapter.score_pairs_torch(image_bchw, texts, output_attentions=False)
    return scores.mean()


def _normalized_grad(grad):
    return grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)


def _make_kernel(kernel_size: int, sigma: float, device: str, channels: int):
    import torch

    size = max(1, int(kernel_size))
    if size % 2 == 0:
        size += 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(x * x + y * y) / (2.0 * float(sigma) * float(sigma)))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, size, size).repeat(int(channels), 1, 1, 1)


def _smooth_grad(grad, kernel):
    import torch.nn.functional as F

    pad = int(kernel.shape[-1] // 2)
    return F.conv2d(grad, kernel, padding=pad, groups=int(grad.shape[1]))


def _input_diversity(x, cfg: Any, rng: np.random.Generator):
    import torch.nn.functional as F

    probability = float(getattr(cfg, "diversity_prob", 0.7) or 0.7)
    if probability <= 0 or float(rng.random()) > probability:
        return x

    _, _, height, width = x.shape
    min_side = min(int(height), int(width))
    max_side = max(int(height), int(width))
    low = max(1, int(round(min_side * float(getattr(cfg, "resize_rate", 0.9) or 0.9))))
    if low >= max_side:
        return x

    target_side = int(rng.integers(low, max_side + 1))
    resized = F.interpolate(x, size=(target_side, target_side), mode="bilinear", align_corners=False)

    pad_h = max(0, height - target_side)
    pad_w = max(0, width - target_side)
    top = int(rng.integers(0, pad_h + 1)) if pad_h > 0 else 0
    left = int(rng.integers(0, pad_w + 1)) if pad_w > 0 else 0
    bottom = pad_h - top
    right = pad_w - left
    return F.pad(resized, (left, right, top, bottom), value=0.0)


def _variance_grad(
    *,
    adapter: Any,
    adv,
    text: str,
    cfg: Any,
    samples: int,
):
    import torch

    if samples <= 0:
        return torch.zeros_like(adv)

    radius = float(getattr(cfg, "variance_radius", 0.05) or 0.05) * max(1e-6, float(getattr(cfg, "epsilon", 0.0) or 0.0))
    if radius <= 0:
        return torch.zeros_like(adv)

    grads = []
    for _ in range(int(samples)):
        noise = (torch.rand_like(adv) * 2.0 - 1.0) * radius
        neighbor = (adv.detach() + noise).clamp(0.0, 1.0).requires_grad_(True)
        loss = _score_loss(adapter, neighbor, text)
        grads.append(torch.autograd.grad(loss, neighbor, retain_graph=False, create_graph=False)[0])
    return torch.stack(grads, dim=0).mean(dim=0)


def _write_debug(sample_debug_dir: str, payload: dict[str, Any]) -> str:
    debug_dir = Path(sample_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    out = debug_dir / "classic_attack_debug.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def _torchattacks_available() -> bool:
    return torchattacks is not None


class _TorchAttackScoringModel:
    def __init__(self, adapter: Any, *, device: str, text: str):
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self, adapter_obj: Any, module_device: str, module_text: str) -> None:
                super().__init__()
                self.adapter = adapter_obj
                self.texts = [module_text]
                self.anchor = nn.Parameter(torch.zeros(1, device=module_device), requires_grad=False)

            def set_texts(self, texts: list[str]) -> None:
                self.texts = [str(text) for text in texts] or [""]

            def forward(self, images):
                scores = self.adapter.score_pairs_torch(images, self.texts * int(images.shape[0]), output_attentions=False)
                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)
                return torch.stack([scores, -scores], dim=1)

        self.module = _Module(adapter, device, text)


class _ClassicGradientAttack(AttackPlugin):
    def __init__(self, spec: ClassicAttackSpec) -> None:
        self.spec = spec

    def _torchattack_kwargs(self, cfg: Any) -> dict[str, Any]:
        epsilon = float(getattr(cfg, "epsilon", 0.05) or 0.05)
        step_size = float(getattr(cfg, "step_size", 0.01) or 0.01)
        steps = max(1, int(self.spec.steps_override or getattr(cfg, "steps", 8) or 8))
        momentum_decay = float(getattr(cfg, "momentum_decay", 1.0) or 1.0)

        if self.spec.key == "fgsm":
            return {"eps": epsilon}
        if self.spec.key == "bim":
            return {"eps": epsilon, "alpha": step_size, "steps": steps}
        if self.spec.key == "pgd":
            return {"eps": epsilon, "alpha": step_size, "steps": steps, "random_start": True}
        if self.spec.key == "mifgsm":
            return {"eps": epsilon, "alpha": step_size, "steps": steps, "decay": momentum_decay}
        if self.spec.key == "nifgsm":
            return {"eps": epsilon, "alpha": step_size, "steps": steps, "decay": momentum_decay}
        if self.spec.key == "difgsm":
            return {
                "eps": epsilon,
                "alpha": step_size,
                "steps": steps,
                "decay": 0.0,
                "resize_rate": float(getattr(cfg, "resize_rate", 0.9) or 0.9),
                "diversity_prob": float(getattr(cfg, "diversity_prob", 0.7) or 0.7),
                "random_start": self.spec.random_start,
            }
        if self.spec.key == "tifgsm":
            return {
                "eps": epsilon,
                "alpha": step_size,
                "steps": steps,
                "decay": 0.0,
                "len_kernel": max(3, int(getattr(cfg, "kernel_size", 5) or 5)),
                "nsig": max(0.1, float(getattr(cfg, "kernel_sigma", 1.0) or 1.0)),
                "resize_rate": float(getattr(cfg, "resize_rate", 0.9) or 0.9),
                "diversity_prob": float(getattr(cfg, "diversity_prob", 0.7) or 0.7),
                "random_start": self.spec.random_start,
            }
        if self.spec.key == "vmifgsm":
            return {
                "eps": epsilon,
                "alpha": step_size,
                "steps": steps,
                "decay": momentum_decay,
                "N": max(1, int(getattr(cfg, "variance_samples", 2) or 2)),
                "beta": max(0.1, float(getattr(cfg, "variance_radius", 0.05) or 0.05)),
            }
        if self.spec.key == "vnifgsm":
            return {
                "eps": epsilon,
                "alpha": step_size,
                "steps": steps,
                "decay": momentum_decay,
                "N": max(1, int(getattr(cfg, "variance_samples", 2) or 2)),
                "beta": max(0.1, float(getattr(cfg, "variance_radius", 0.05) or 0.05)),
            }
        if self.spec.key == "cw":
            return {
                "c": float(getattr(cfg, "cw_const", 0.1) or 0.1),
                "kappa": float(getattr(cfg, "cw_confidence", 0.1) or 0.1),
                "steps": steps,
                "lr": step_size,
            }
        return {}

    def _build_torchattack(self, *, adapter: Any, cfg: Any, text: str, device: str):
        if not (_torchattacks_available() and self.spec.torchattack_name):
            return None

        attack_cls = getattr(torchattacks, str(self.spec.torchattack_name), None)
        if attack_cls is None:
            return None

        scoring_model = _TorchAttackScoringModel(adapter, device=device, text=text)
        attacker = attack_cls(scoring_model.module, **self._torchattack_kwargs(cfg))
        try:
            attacker.set_device(device)
            setattr(attacker, "_mmsec_set_device_status", "ready")
        except (AttributeError, TypeError, RuntimeError) as exc:
            setattr(attacker, "_mmsec_set_device_status", f"skipped:{type(exc).__name__}")
        return attacker, scoring_model.module

    def _run_with_torchattacks(self, *, clean, text: str, adapter: Any, cfg: Any, device: str) -> tuple[np.ndarray, list[AttackTraceStep], dict[str, Any]]:
        import torch

        built = self._build_torchattack(adapter=adapter, cfg=cfg, text=text, device=device)
        if built is None:
            raise RuntimeError("torchattacks is not available for this attack")
        attacker, scoring_model = built
        labels = torch.zeros((int(clean.shape[0]),), dtype=torch.long, device=clean.device)

        with torch.no_grad():
            score_before = float(_score_loss(adapter, clean, text).detach().cpu().item())

        adv = attacker(clean.detach(), labels).detach().clamp(0.0, 1.0)

        with torch.no_grad():
            score_after = float(_score_loss(adapter, adv, text).detach().cpu().item())
            mean_abs_delta = float((adv - clean).abs().mean().detach().cpu().item())

        traces = [
            AttackTraceStep(
                step=1,
                loss_total=score_after,
                loss_parts={
                    "score_before": score_before,
                    "score_after": score_after,
                    "score_drop": score_before - score_after,
                },
                metadata={
                    "implementation": "torchattacks",
                    "attack_class": str(self.spec.torchattack_name),
                    "mean_abs_delta": mean_abs_delta,
                    "set_device_status": str(getattr(attacker, "_mmsec_set_device_status", "unknown")),
                },
            )
        ]
        adv_np = adv[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        return adv_np, traces, {
            "implementation": "torchattacks",
            "attack_class": str(self.spec.torchattack_name),
            "set_device_status": str(getattr(attacker, "_mmsec_set_device_status", "unknown")),
        }

    def _run_cw(self, *, clean, text: str, adapter: Any, cfg: Any) -> tuple[np.ndarray, list[AttackTraceStep], dict[str, Any]]:
        import torch

        step_size = float(getattr(cfg, "step_size", 0.01) or 0.01)
        steps = max(1, int(self.spec.steps_override or getattr(cfg, "steps", 8) or 8))
        epsilon = float(getattr(cfg, "epsilon", 0.05) or 0.05)
        cw_const = float(getattr(cfg, "cw_const", 0.1) or 0.1)
        confidence = float(getattr(cfg, "cw_confidence", 0.1) or 0.1)

        delta = torch.zeros_like(clean, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=step_size)
        traces: list[AttackTraceStep] = []

        for idx in range(steps):
            adv = (clean + delta).clamp(0.0, 1.0)
            score = _score_loss(adapter, adv, text)
            l2 = ((adv - clean) ** 2).reshape(adv.shape[0], -1).sum(dim=1).mean()
            hinge = torch.relu(score - confidence)
            loss = hinge + cw_const * l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)

            grad_norm = float(delta.grad.detach().abs().mean().cpu().item()) if delta.grad is not None else 0.0
            traces.append(
                AttackTraceStep(
                    step=idx + 1,
                    loss_total=float(loss.detach().cpu().item()),
                    loss_parts={
                        "score": float(score.detach().cpu().item()),
                        "hinge": float(hinge.detach().cpu().item()),
                        "l2": float(l2.detach().cpu().item()),
                    },
                    metadata={"grad_mean_abs": grad_norm, "implementation": "builtin"},
                )
            )

        adv = (clean + delta.detach()).clamp(0.0, 1.0)
        adv_np = adv[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        return adv_np, traces, {"implementation": "builtin", "attack_class": "custom_cw"}

    def _make_iterative_trace(self, *, idx: int, loss: Any, grad: Any, momentum: Any) -> AttackTraceStep:
        return AttackTraceStep(
            step=idx + 1,
            loss_total=float(loss.detach().cpu().item()),
            loss_parts={
                "score": float(loss.detach().cpu().item()),
                "grad_mean_abs": float(grad.detach().abs().mean().cpu().item()),
                "momentum_mean_abs": float(momentum.detach().abs().mean().cpu().item()) if self.spec.use_momentum else 0.0,
            },
            metadata={
                "implementation": "builtin",
                "random_start": bool(self.spec.random_start),
                "input_diversity": bool(self.spec.input_diversity),
                "translation_invariant": bool(self.spec.translation_invariant),
                "variance_tuning": bool(self.spec.variance_tuning),
                "nesterov": bool(self.spec.nesterov),
            },
        )

    def _shape_grouped_attack_batch(self, samples: list[Sample], ctx: AttackContext) -> list[AttackedSample]:
        grouped: dict[tuple[int, ...], list[tuple[int, Sample]]] = {}
        for idx, sample in enumerate(samples):
            grouped.setdefault(tuple(np.asarray(sample.image).shape), []).append((idx, sample))
        ordered: list[AttackedSample | None] = [None] * len(samples)
        for grouped_items in grouped.values():
            indices = [idx for idx, _sample in grouped_items]
            grouped_samples = [_sample for _idx, _sample in grouped_items]
            attacked_group = self.attack_batch(grouped_samples, ctx)
            if len(attacked_group) != len(grouped_samples):
                raise RuntimeError("shape-grouped batch attack output length mismatch")
            for idx, attacked in zip(indices, attacked_group):
                attacked.metadata["shape_grouped_batch"] = True
                ordered[idx] = attacked
        if any(item is None for item in ordered):
            raise RuntimeError("shape-grouped batch attack left empty outputs")
        return [item for item in ordered if item is not None]

    def _batch_sample(self, *, sample: Sample, adv_np: np.ndarray, mode: str, traces: list[AttackTraceStep], implementation_meta: dict[str, Any], cfg: Any, batch_size: int) -> AttackedSample:
        l0, l2, linf = perturb_stats(np.asarray(sample.image, dtype=np.float32), adv_np)
        adv_sample = Sample(
            sample_id=sample.sample_id,
            image=adv_np,
            text=str(sample.text),
            target_text=str(sample.target_text or ""),
            metadata=dict(sample.metadata),
        )
        adv_sample.metadata.update(
            {
                "attack_name": self.spec.key,
                "attack_mode": mode,
                "attack_scope": "image",
                "attack_family": "classic_gradient",
                "attack_implementation": "builtin_batch",
            }
        )
        metadata = self._single_metadata(cfg=cfg, debug_path="", implementation_meta={"implementation": "builtin_batch", **implementation_meta})
        metadata["batch_size"] = int(batch_size)
        return AttackedSample(sample=adv_sample, perturbation_l0=l0, perturbation_l2=l2, perturbation_linf=linf, attack_trace=traces, metadata=metadata)

    def _run_iterative(
        self,
        *,
        clean,
        text: str,
        adapter: Any,
        cfg: Any,
        sample_id: str,
        mode: str,
    ) -> tuple[np.ndarray, list[AttackTraceStep], dict[str, Any]]:
        import torch

        rng = np.random.default_rng(_seed(sample_id, self.spec.key, mode))
        epsilon = float(getattr(cfg, "epsilon", 0.05) or 0.05)
        step_size = float(getattr(cfg, "step_size", 0.01) or 0.01)
        steps = max(1, int(self.spec.steps_override or getattr(cfg, "steps", 8) or 8))
        momentum_decay = float(getattr(cfg, "momentum_decay", 1.0) or 1.0)
        nesterov_scale = float(getattr(cfg, "nesterov_scale", 1.0) or 1.0)

        adv = clean.detach().clone()
        if self.spec.random_start:
            noise = torch.from_numpy(rng.uniform(-epsilon, epsilon, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
            adv = (adv + noise).clamp(0.0, 1.0)

        momentum = torch.zeros_like(clean)
        kernel = None
        if self.spec.translation_invariant:
            kernel = _make_kernel(
                kernel_size=int(getattr(cfg, "kernel_size", 5) or 5),
                sigma=float(getattr(cfg, "kernel_sigma", 1.0) or 1.0),
                device=str(clean.device),
                channels=int(clean.shape[1]),
            )

        traces: list[AttackTraceStep] = []
        for idx in range(steps):
            current = adv.detach().clone().requires_grad_(True)
            attack_input = current
            if self.spec.nesterov and self.spec.use_momentum:
                lookahead = current - step_size * nesterov_scale * momentum_decay * momentum.sign()
                attack_input = lookahead.clamp(0.0, 1.0)
            if self.spec.input_diversity:
                attack_input = _input_diversity(attack_input, cfg, rng)

            loss = _score_loss(adapter, attack_input, text)
            grad = torch.autograd.grad(loss, current, retain_graph=False, create_graph=False)[0]

            if self.spec.variance_tuning:
                grad = grad + _variance_grad(
                    adapter=adapter,
                    adv=current,
                    text=text,
                    cfg=cfg,
                    samples=int(getattr(cfg, "variance_samples", 2) or 0),
                )
            if kernel is not None:
                grad = _smooth_grad(grad, kernel)

            grad = _normalized_grad(grad)
            if self.spec.use_momentum:
                momentum = momentum_decay * momentum + grad
                update = momentum
            else:
                update = grad

            next_adv = current - step_size * update.sign()
            delta = (next_adv - clean).clamp(-epsilon, epsilon)
            adv = (clean + delta).clamp(0.0, 1.0).detach()
            traces.append(self._make_iterative_trace(idx=idx, loss=loss, grad=grad, momentum=momentum))

        adv_np = adv.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
        return adv_np, traces, {"implementation": "builtin", "attack_class": "custom_iterative"}

    def _run_single_variant(
        self,
        *,
        clean: Any,
        text: str,
        adapter: Any,
        cfg: Any,
        device: str,
        sample_id: str,
        mode: str,
    ) -> tuple[np.ndarray, list[AttackTraceStep], dict[str, Any], str]:
        fallback_error = ""
        if self.spec.torchattack_name and _torchattacks_available():
            try:
                adv_np, traces, implementation_meta = self._run_with_torchattacks(clean=clean, text=text, adapter=adapter, cfg=cfg, device=device)
                return adv_np, traces, implementation_meta, fallback_error
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                fallback_error = str(exc)
        if self.spec.cw_style:
            adv_np, traces, implementation_meta = self._run_cw(clean=clean, text=text, adapter=adapter, cfg=cfg)
        else:
            adv_np, traces, implementation_meta = self._run_iterative(
                clean=clean,
                text=text,
                adapter=adapter,
                cfg=cfg,
                sample_id=sample_id,
                mode=mode,
            )
        return adv_np, traces, implementation_meta, fallback_error

    def _debug_payload(
        self,
        *,
        cfg: Any,
        mode: str,
        traces: list[AttackTraceStep],
        implementation_meta: dict[str, Any],
        fallback_error: str,
    ) -> dict[str, Any]:
        return {
            "attack": self.spec.key,
            "mode": mode,
            "epsilon": float(getattr(cfg, "epsilon", 0.0) or 0.0),
            "step_size": float(getattr(cfg, "step_size", 0.0) or 0.0),
            "steps": int(self.spec.steps_override or getattr(cfg, "steps", 0) or 0),
            "random_start": bool(self.spec.random_start),
            "use_momentum": bool(self.spec.use_momentum),
            "nesterov": bool(self.spec.nesterov),
            "input_diversity": bool(self.spec.input_diversity),
            "translation_invariant": bool(self.spec.translation_invariant),
            "variance_tuning": bool(self.spec.variance_tuning),
            "cw_style": bool(self.spec.cw_style),
            "trace_steps": len(traces),
            "implementation": implementation_meta.get("implementation", "builtin"),
            "attack_class": implementation_meta.get("attack_class", ""),
            "fallback_error": fallback_error,
            "trace_tail": [
                {
                    "step": t.step,
                    "loss_total": t.loss_total,
                    "loss_parts": t.loss_parts,
                    "metadata": t.metadata,
                }
                for t in traces[-3:]
            ],
        }

    def _single_metadata(self, *, cfg: Any, debug_path: str, implementation_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "variant": self.spec.key.upper(),
            "scope": "image",
            "debug_path": debug_path,
            "implementation": implementation_meta.get("implementation", "builtin"),
            "attack_class": implementation_meta.get("attack_class", ""),
            "parameters": {
                "epsilon": float(getattr(cfg, "epsilon", 0.0) or 0.0),
                "step_size": float(getattr(cfg, "step_size", 0.0) or 0.0),
                "steps": int(self.spec.steps_override or getattr(cfg, "steps", 0) or 0),
            },
        }

    def attack_batch(self, samples: list[Sample], ctx: AttackContext) -> list[AttackedSample]:
        if not samples:
            return []
        if self.spec.cw_style:
            return [self.attack(sample, ctx) for sample in samples]

        if len({tuple(np.asarray(sample.image).shape) for sample in samples}) > 1:
            return self._shape_grouped_attack_batch(samples, ctx)

        device = _require_gradient_adapter(ctx.surrogate_model_adapter or ctx.model_adapter)
        import torch

        clean = torch.cat([_to_tensor(sample.image, device) for sample in samples], dim=0)
        cfg = ctx.config.attack
        mode = str(getattr(cfg, "mode", "A") or "A").upper()
        adapter = ctx.surrogate_model_adapter or ctx.model_adapter
        texts = [str(sample.text or "") for sample in samples]
        batch_id = "|".join(str(sample.sample_id) for sample in samples[:8])
        adv_np_batch, traces, implementation_meta = self._run_iterative(
            clean=clean,
            text=texts,
            adapter=adapter,
            cfg=cfg,
            sample_id=batch_id,
            mode=mode,
        )
        if adv_np_batch.ndim != 4 or adv_np_batch.shape[0] != len(samples):
            raise RuntimeError(f"invalid batch attack output shape: {adv_np_batch.shape}")

        return [
            self._batch_sample(sample=sample, adv_np=adv_np_batch[idx], mode=mode, traces=traces, implementation_meta=implementation_meta, cfg=cfg, batch_size=len(samples))
            for idx, sample in enumerate(samples)
        ]

    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        device = _require_gradient_adapter(ctx.surrogate_model_adapter or ctx.model_adapter)
        clean = _to_tensor(sample.image, device)
        cfg = ctx.config.attack
        mode = str(getattr(cfg, "mode", "A") or "A").upper()
        adapter = ctx.surrogate_model_adapter or ctx.model_adapter
        text = str(sample.text or "")

        adv_np, traces, implementation_meta, fallback_error = self._run_single_variant(
            clean=clean,
            text=text,
            adapter=adapter,
            cfg=cfg,
            device=device,
            sample_id=sample.sample_id,
            mode=mode,
        )
        if getattr(adv_np, "ndim", 0) == 4:
            adv_np = adv_np[0]

        l0, l2, linf = perturb_stats(np.asarray(sample.image, dtype=np.float32), adv_np)
        adv_sample = Sample(
            sample_id=sample.sample_id,
            image=adv_np,
            text=str(sample.text),
            target_text=str(sample.target_text or ""),
            metadata=dict(sample.metadata),
        )
        adv_sample.metadata["attack_name"] = self.spec.key
        adv_sample.metadata["attack_mode"] = mode
        adv_sample.metadata["attack_scope"] = "image"
        adv_sample.metadata["attack_family"] = "classic_gradient"
        adv_sample.metadata["attack_implementation"] = str(implementation_meta.get("implementation", "builtin"))

        debug_path = ""
        if ctx.sample_debug_dir:
            debug_path = _write_debug(
                ctx.sample_debug_dir,
                self._debug_payload(cfg=cfg, mode=mode, traces=traces, implementation_meta=implementation_meta, fallback_error=fallback_error),
            )

        return AttackedSample(
            sample=adv_sample,
            perturbation_l0=l0,
            perturbation_l2=l2,
            perturbation_linf=linf,
            attack_trace=traces,
            metadata=self._single_metadata(cfg=cfg, debug_path=debug_path, implementation_meta=implementation_meta),
        )


class FGSMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="fgsm", steps_override=1, torchattack_name="FGSM"))


class BIMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="bim", torchattack_name="BIM"))


class PGDAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="pgd", random_start=True, torchattack_name="PGD"))


class MIFGSMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="mifgsm", use_momentum=True, torchattack_name="MIFGSM"))


class NIFGSMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="nifgsm", use_momentum=True, nesterov=True, torchattack_name="NIFGSM"))


class DIFGSMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="difgsm", input_diversity=True, torchattack_name="DIFGSM"))


class TIFGSMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="tifgsm", translation_invariant=True, torchattack_name="TIFGSM"))


class DTMIFGSMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(
            ClassicAttackSpec(
                key="dtmifgsm",
                use_momentum=True,
                input_diversity=True,
                translation_invariant=True,
            )
        )


class VMIFGSMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="vmifgsm", use_momentum=True, variance_tuning=True, torchattack_name="VMIFGSM"))


class VNIFGSMAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(
            ClassicAttackSpec(
                key="vnifgsm",
                use_momentum=True,
                nesterov=True,
                variance_tuning=True,
                torchattack_name="VNIFGSM",
            )
        )


class CWAttack(_ClassicGradientAttack):
    def __init__(self) -> None:
        super().__init__(ClassicAttackSpec(key="cw", cw_style=True, torchattack_name="CW"))
