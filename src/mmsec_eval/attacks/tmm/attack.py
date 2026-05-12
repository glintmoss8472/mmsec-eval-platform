# 文件说明：该文件属于TMM 迁移攻击模块，集中实现 attack 相关逻辑。
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mmsec_eval.attacks.base import perturb_stats
from mmsec_eval.attacks.text_utils import run_text_replacement_attack
from mmsec_eval.attacks.tmm.adfp import allocate_budget, attention_to_critical_mask, input_diversity
from mmsec_eval.attacks.tmm.losses import ogfh_lo, ssim_loss
from mmsec_eval.attacks.tmm.ogfh import orthogonalize_grad
from mmsec_eval.plugins.base import AttackPlugin
from mmsec_eval.types import AttackContext, AttackTraceStep, AttackedSample, Sample


# 中文注释：封装 _seed 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
def _seed(sample_id: str, mode: str) -> int:
    return int(hashlib.sha256(f"tmm:{sample_id}:{mode}".encode("utf-8")).hexdigest(), 16) % (2**31 - 1)


# 中文注释：封装 _tmm_text_attack 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
def _tmm_text_attack(
    *,
    image: np.ndarray,
    text: str,
    adapter: Any,
    eps_t: int,
    candidates_k: int,
) -> tuple[str, dict[str, Any]]:
    return run_text_replacement_attack(
        image=image,
        text=text,
        adapter=adapter,
        eps_t=int(eps_t),
        candidates_k=int(candidates_k),
        prefer_mlm=True,
    )


# 中文注释：封装 _tmm_scope 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
def _tmm_scope(cfg: Any) -> tuple[str, bool, bool]:
    scope = str(getattr(cfg.task, "eval_scope", "joint") or "joint")
    if str(getattr(cfg.task, "kind", "pairwise")) != "vlr" or scope == "clean":
        scope = "joint"
    return scope, scope in {"image", "joint"}, scope in {"text", "joint"}


# 中文注释：封装 _attention_budget 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
def _attention_budget(clean: np.ndarray, text: str, ctx: AttackContext, acfg: Any) -> tuple[np.ndarray, np.ndarray, float, float]:
    att_map = None
    try:
        if hasattr(ctx.model_adapter, "attention_map"):
            att_map = ctx.model_adapter.attention_map(clean, text)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        att_map = None
    if att_map is None:
        raise RuntimeError("TMM requires model_adapter.attention_map")
    crit_mask = attention_to_critical_mask(att_map, float(acfg.lambda_att))
    eps_crit, eps_non = allocate_budget(crit_mask, float(acfg.epsilon), float(acfg.ratio_r))
    return att_map, crit_mask, eps_crit, eps_non


# 中文注释：定义 TMMAttack 的结构化职责，作为TMM 迁移攻击模块中状态、配置或行为的边界。
class TMMAttack(AttackPlugin):
    """Transferable Multimodal Attack (TMM), image+text.

    Implements a practical, paper-aligned pipeline:
      - ADFP: attention-directed critical mask + budget split
      - OGFH: orthogonal-guided gradient shaping (simplified)
      - δt: token replacement via BERT MLM (eps_t=1 by default)

    In pairwise (task.kind=pairwise) runs, we default to joint attacks to keep legacy behavior.
    In VLR (task.kind=vlr) runs, we respect task.eval_scope (image/text/joint).
    """

    # 中文注释：封装 TMMAttack._run_image_branch 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
    def _run_image_branch(self, *, sample_id: str, mode: str, scope: str, clean: np.ndarray, text: str, ctx: AttackContext, crit_mask: np.ndarray, eps_crit: float, eps_non: float) -> tuple[np.ndarray, list[AttackTraceStep]]:
        if not hasattr(ctx.model_adapter, "score_pairs_torch") or not hasattr(ctx.model_adapter, "projected_features_torch"):
            raise RuntimeError("TMM image attack requires torch scoring and projected features")
        import torch

        acfg = ctx.config.attack
        device = getattr(ctx.model_adapter, "_device", "cpu")
        x0 = torch.from_numpy(clean).permute(2, 0, 1).unsqueeze(0).to(device)
        adv = x0.clone().detach()
        with torch.no_grad():
            img_feat_clean, txt_feat_clean = ctx.model_adapter.projected_features_torch(x0, [text])
        eps_map = self._eps_map(crit_mask=crit_mask, eps_crit=eps_crit, eps_non=eps_non, clean=clean, device=device)
        rng = np.random.default_rng(_seed(sample_id, mode))
        momentum = torch.zeros_like(adv)
        traces: list[AttackTraceStep] = []
        for k in range(max(1, int(acfg.steps))):
            adv, momentum, trace = self._image_step(
                adv=adv,
                x0=x0,
                text=text,
                ctx=ctx,
                rng=rng,
                momentum=momentum,
                eps_map=eps_map,
                img_feat_clean=img_feat_clean,
                txt_feat_clean=txt_feat_clean,
                step=k,
                mode=mode,
                scope=scope,
            )
            traces.append(trace)
        return adv[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32), traces

    # 中文注释：封装 TMMAttack._eps_map 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
    def _eps_map(self, *, crit_mask: np.ndarray, eps_crit: float, eps_non: float, clean: np.ndarray, device: Any):
        import torch

        m = torch.from_numpy(crit_mask).to(device=device, dtype=torch.float32).view(1, 1, clean.shape[0], clean.shape[1])
        return (eps_non * (1.0 - m) + eps_crit * m).expand(1, 3, clean.shape[0], clean.shape[1])

    # 中文注释：封装 TMMAttack._image_step 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
    def _image_step(self, *, adv: Any, x0: Any, text: str, ctx: AttackContext, rng: Any, momentum: Any, eps_map: Any, img_feat_clean: Any, txt_feat_clean: Any, step: int, mode: str, scope: str):
        import torch

        acfg = ctx.config.attack
        adv = adv.detach().clone().requires_grad_(True)
        adv_in = input_diversity(adv, rng=rng, p=0.7, low=0.9)
        score = ctx.model_adapter.score_pairs_torch(adv_in, [text], output_attentions=False).mean()
        img_feat_adv, txt_feat_adv = ctx.model_adapter.projected_features_torch(adv_in, [text])
        lo, lo_parts = ogfh_lo(img_feat_adv=img_feat_adv, img_feat_clean=img_feat_clean, txt_feat=txt_feat_adv, txt_feat_clean=txt_feat_clean)
        ls = ssim_loss(x0, adv, window=11)
        total = float((acfg.loss_weights or {}).get("target", 1.0)) * score + lo + float(acfg.alpha) * ls
        grad = torch.autograd.grad(total, adv, retain_graph=False, create_graph=False)[0]
        grad = orthogonalize_grad(grad, x0)
        grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
        momentum = float(acfg.gamma if float(acfg.gamma) > 0 else 1.0) * momentum + grad
        delta = (adv - float(acfg.step_size) * momentum.sign() - x0).clamp(-eps_map, eps_map)
        next_adv = (x0 + delta).clamp(0.0, 1.0)
        return next_adv, momentum, self._trace(step, total, score, lo, ls, lo_parts, momentum, mode, scope, eps_map)

    # 中文注释：封装 TMMAttack._trace 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
    def _trace(self, step: int, total: Any, score: Any, lo: Any, ls: Any, lo_parts: dict[str, Any], momentum: Any, mode: str, scope: str, eps_map: Any) -> AttackTraceStep:
        return AttackTraceStep(
            step=step + 1,
            loss_total=float(total.detach().cpu().item()),
            loss_parts={
                "score": float(score.detach().cpu().item()),
                "LO": float(lo.detach().cpu().item()),
                "LS": float(ls.detach().cpu().item()),
                "Lov": float(lo_parts.get("Lov", lo).detach().cpu().item()),
                "Lot": float(lo_parts.get("Lot", lo).detach().cpu().item()),
                "Loa": float(lo_parts.get("Loa", lo).detach().cpu().item()),
                "momentum_l1": float(momentum.abs().mean().detach().cpu().item()),
            },
            metadata={"mode": mode, "scope": scope, "eps_crit": float(eps_map.max().detach().cpu().item()), "eps_non": float(eps_map.min().detach().cpu().item())},
        )

    # 中文注释：实现 TMMAttack.attack 的核心行为，维护TMM 迁移攻击模块在该对象上的调用契约。
    def attack(self, sample: Sample, ctx: AttackContext) -> AttackedSample:
        acfg = ctx.config.attack
        mode = str(acfg.mode).upper()
        scope, do_img, do_txt = _tmm_scope(ctx.config)
        clean = np.asarray(sample.image, dtype=np.float32)
        text = str(sample.text or "")
        att_map, crit_mask, eps_crit, eps_non = _attention_budget(clean, text, ctx, acfg)
        adv_img = clean
        traces: list[AttackTraceStep] = []
        if do_img:
            adv_img, traces = self._run_image_branch(
                sample_id=sample.sample_id,
                mode=mode,
                scope=scope,
                clean=clean,
                text=text,
                ctx=ctx,
                crit_mask=crit_mask,
                eps_crit=eps_crit,
                eps_non=eps_non,
            )

        # Text attack (δt).
        text_edit: dict[str, Any] = {"method": "noop"}
        if do_txt:
            adv_text, text_edit = _tmm_text_attack(
                image=adv_img if do_img else clean,
                text=text,
                adapter=ctx.model_adapter,
                eps_t=int(acfg.eps_t),
                candidates_k=int(acfg.text_candidates_k),
            )

        # Scope enforcement.
        if not do_img:
            adv_img = clean
        if not do_txt:
            adv_text = text

        l0, l2, linf = perturb_stats(clean, adv_img)
        adv_sample = _make_tmm_sample(sample, adv_img, adv_text, mode, scope)
        debug_path = _write_tmm_debug(
            sample_debug_dir=ctx.sample_debug_dir,
            att_map=att_map,
            crit_mask=crit_mask,
            eps_crit=eps_crit,
            eps_non=eps_non,
            text_edit=text_edit,
            traces=traces,
        ) if ctx.sample_debug_dir else ""

        return AttackedSample(
            sample=adv_sample,
            perturbation_l0=l0,
            perturbation_l2=l2,
            perturbation_linf=linf,
            attack_trace=traces,
            metadata={
                "mode": mode,
                "scope": scope,
                "critical_ratio": float(crit_mask.mean()),
                "eps_crit": float(eps_crit),
                "eps_non": float(eps_non),
                "text_edit": text_edit,
                "debug_path": debug_path,
            },
        )


# 中文注释：封装 _make_tmm_sample 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
def _make_tmm_sample(sample: Sample, adv_img: np.ndarray, adv_text: str, mode: str, scope: str) -> Sample:
    adv_sample = Sample(
        sample_id=sample.sample_id,
        image=adv_img,
        text=adv_text,
        target_text=str(sample.target_text or ""),
        metadata=dict(sample.metadata),
    )
    adv_sample.metadata.update({"attack_name": "tmm", "attack_mode": mode, "attack_scope": scope})
    return adv_sample


# 中文注释：封装 _write_tmm_debug 的内部步骤，让TMM 迁移攻击模块主流程保持清晰并隔离边界细节。
def _write_tmm_debug(
    *,
    sample_debug_dir: str,
    att_map,
    crit_mask,
    eps_crit: float,
    eps_non: float,
    text_edit: dict[str, Any],
    traces: list[AttackTraceStep],
) -> str:
    debug_dir = Path(sample_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    out = debug_dir / "tmm_debug.json"
    payload = {
        "att_map_mean": float(np.asarray(att_map).mean()),
        "att_map_std": float(np.asarray(att_map).std()),
        "critical_ratio": float(np.asarray(crit_mask).mean()),
        "eps_crit": float(eps_crit),
        "eps_non": float(eps_non),
        "text_edit": text_edit,
        "trace_steps": len(traces),
        "trace_tail": [
            {"step": t.step, "loss_total": t.loss_total, "loss_parts": t.loss_parts, "metadata": t.metadata}
            for t in traces[-3:]
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)
