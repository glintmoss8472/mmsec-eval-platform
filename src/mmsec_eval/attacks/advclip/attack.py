# 文件说明：该文件属于AdvCLIP 攻击模块，集中实现 attack 相关逻辑。
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from mmsec_eval.attacks.advclip.patch import apply_patch, clamp_patch, patch_tv, random_location
from mmsec_eval.attacks.advclip.registry import make_key, resolve_patch
from mmsec_eval.attacks.base import perturb_stats
from mmsec_eval.plugins.base import AttackPlugin
from mmsec_eval.types import AttackContext, AttackTraceStep, AttackedSample, Sample


# 中文注释：定义 _PatchResolution 的结构化职责，作为AdvCLIP 攻击模块中状态、配置或行为的边界。
@dataclass(frozen=True)
class _PatchResolution:
    patch_path: str
    patch_source: str
    registry_key: str


# 中文注释：定义 AdvCLIPPatchAttack 的结构化职责，作为AdvCLIP 攻击模块中状态、配置或行为的边界。
class AdvCLIPPatchAttack(AttackPlugin):
    """AdvCLIP-inspired universal patch attack (apply-time).

    Training of the universal patch is handled by `mmsec_eval train-advclip`.
    This plugin applies a saved patch to produce adversarial images for evaluation.
    """

    # 中文注释：封装 AdvCLIPPatchAttack.__init__ 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        self._patch_cache: dict[tuple[str, int], np.ndarray] = {}
        self._patch_origin: dict[tuple[str, int], dict[str, str]] = {}

    # 中文注释：实现 AdvCLIPPatchAttack.save_patch 的核心行为，维护AdvCLIP 攻击模块在该对象上的调用契约。
    def save_patch(self, mode: str, patch_size: int, path: str) -> str:
        key = (mode.upper(), int(patch_size))
        if key not in self._patch_cache:
            raise ValueError(f"patch not available for key={key}")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, self._patch_cache[key])
        return str(out)

    # 中文注释：实现 AdvCLIPPatchAttack.load_patch 的核心行为，维护AdvCLIP 攻击模块在该对象上的调用契约。
    def load_patch(self, mode: str, patch_size: int, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        arr = np.load(p)
        key = (mode.upper(), int(patch_size))
        self._patch_cache[key] = np.asarray(arr, dtype=np.float32)
        return True

    # 中文注释：封装 AdvCLIPPatchAttack._run_patch_path 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
    def _run_patch_path(self, ctx: AttackContext, *, mode: str, patch_size: int) -> str:
        if ctx.run_dir:
            return str(Path(ctx.run_dir) / "attack_debug" / f"advclip_patch_{mode}_{patch_size}.npy")
        return ""

    # 中文注释：封装 AdvCLIPPatchAttack._resolve_patch 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
    def _resolve_patch(
        self,
        *,
        key: tuple[str, int],
        mode: str,
        patch_size: int,
        ctx: AttackContext,
        patch_path: str,
    ) -> _PatchResolution:
        origin = self._patch_origin.get(key, {})
        patch_source = origin.get("patch_source", "")
        registry_key = origin.get("registry_key", "")
        if key in self._patch_cache:
            return _PatchResolution(patch_path=patch_path, patch_source=str(patch_source), registry_key=str(registry_key))

        if patch_path and Path(patch_path).exists() and self.load_patch(mode, patch_size, patch_path):
            patch_source = "run_dir"
            registry_key = ""
        else:
            loaded, patch_source, registry_key = self._resolve_registry_patch(
                mode=mode,
                patch_size=patch_size,
                ctx=ctx,
                patch_path=patch_path,
            )
            if not loaded:
                raise RuntimeError(
                    "AdvCLIP patch not found (run_dir and registry lookup both failed). "
                    "Run `mmsec_eval train-advclip` first or provide a patch under run_dir/attack_debug."
                )

        self._patch_origin[key] = {"patch_source": str(patch_source), "registry_key": str(registry_key)}
        return _PatchResolution(patch_path=patch_path, patch_source=str(patch_source), registry_key=str(registry_key))

    # 中文注释：封装 AdvCLIPPatchAttack._resolve_registry_patch 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
    def _resolve_registry_patch(
        self,
        *,
        mode: str,
        patch_size: int,
        ctx: AttackContext,
        patch_path: str,
    ) -> tuple[bool, str, str]:
        try:
            reg_key = make_key(
                clip_model_name=str(ctx.config.model.clip_model_name),
                mode=mode,
                patch_size=patch_size,
            )
            resolved = resolve_patch(str(ctx.config.artifacts_dir), reg_key)
            if not resolved or not self.load_patch(mode, patch_size, resolved):
                return False, "", ""
            if patch_path:
                Path(patch_path).parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copyfile(resolved, patch_path)
                except OSError:
                    self.save_patch(mode, patch_size, patch_path)
            return True, "registry", reg_key
        except (OSError, ValueError, RuntimeError):
            return False, "", ""

    # 中文注释：封装 AdvCLIPPatchAttack._persist_patch_for_run 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
    def _persist_patch_for_run(self, *, mode: str, patch_size: int, patch_path: str) -> tuple[str, str]:
        if not patch_path:
            return "", ""
        try:
            return self.save_patch(mode, patch_size, patch_path), ""
        except (OSError, ValueError) as exc:
            return "", f"{type(exc).__name__}: {exc}"

    # 中文注释：封装 AdvCLIPPatchAttack._make_adv_sample 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
    def _make_adv_sample(self, sample: Sample, adv: np.ndarray, *, mode: str) -> Sample:
        adv_sample = Sample(
            sample_id=sample.sample_id,
            image=adv,
            text=str(sample.text),
            target_text=str(sample.target_text or ""),
            metadata=dict(sample.metadata),
        )
        adv_sample.metadata["attack_name"] = "advclip"
        adv_sample.metadata["attack_mode"] = mode
        return adv_sample

    # 中文注释：实现 AdvCLIPPatchAttack.attack 的核心行为，维护AdvCLIP 攻击模块在该对象上的调用契约。
    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        cfg = ctx.config.attack
        mode = str(cfg.mode).upper()
        patch_size = int(cfg.patch_size)
        key = (mode, patch_size)
        patch_path = self._run_patch_path(ctx, mode=mode, patch_size=patch_size)
        patch_info = self._resolve_patch(key=key, mode=mode, patch_size=patch_size, ctx=ctx, patch_path=patch_path)
        patch = clamp_patch(self._patch_cache[key])
        image = np.asarray(sample.image, dtype=np.float32)

        rng = np.random.default_rng(_seed(sample.sample_id, mode))
        loc = random_location(image.shape, patch_size, rng=rng, margin=8)
        adv = apply_patch(image, patch, loc)

        l0, l2, linf = perturb_stats(image, adv)
        traces = [
            AttackTraceStep(
                step=1,
                loss_total=float(patch_tv(patch)),
                loss_parts={"tv": float(patch_tv(patch))},
                metadata={"mode": mode, "patch_size": patch_size, "position": [int(loc[0]), int(loc[1])]},
            )
        ]

        patch_path, patch_save_error = self._persist_patch_for_run(
            mode=mode,
            patch_size=patch_size,
            patch_path=patch_info.patch_path,
        )

        preview_path = ""
        payload_path = ""
        if ctx.sample_debug_dir:
            preview_path, payload_path = _write_patch_debug(ctx.sample_debug_dir, patch, traces)

        return AttackedSample(
            sample=self._make_adv_sample(sample, adv, mode=mode),
            perturbation_l0=l0,
            perturbation_l2=l2,
            perturbation_linf=linf,
            attack_trace=traces,
            metadata={
                "patch_size": patch_size,
                "mode": mode,
                "position": [int(loc[0]), int(loc[1])],
                "patch_path": patch_path,
                "patch_source": patch_info.patch_source or "unknown",
                "registry_key": patch_info.registry_key or "",
                "patch_preview": preview_path,
                "debug_payload": payload_path,
                "patch_save_error": patch_save_error,
            },
        )


# 中文注释：封装 _seed 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _seed(sample_id: str, mode: str) -> int:
    return int(hashlib.sha256(f"advclip:{sample_id}:{mode}".encode("utf-8")).hexdigest(), 16) % (2**31 - 1)


# 中文注释：封装 _write_patch_debug 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _write_patch_debug(sample_debug_dir: str, patch: np.ndarray, traces: list[AttackTraceStep]) -> tuple[str, str]:
    debug_dir = Path(sample_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    preview = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
    preview_path = debug_dir / "advclip_patch_preview.png"
    Image.fromarray(preview).save(preview_path)

    payload_path = debug_dir / "advclip_patch_debug.json"
    payload = {
        "shape": list(patch.shape),
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "trace_steps": len(traces),
        "loss_tail": [
            {"step": t.step, "loss_total": t.loss_total, "loss_parts": t.loss_parts}
            for t in traces[-3:]
        ],
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(preview_path), str(payload_path)
