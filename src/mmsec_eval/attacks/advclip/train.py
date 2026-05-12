# 文件说明：该文件属于AdvCLIP 攻击模块，集中实现 train 相关逻辑。
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mmsec_eval.attacks.advclip.registry import make_key, update_entry
from mmsec_eval.attacks.advclip.losses import contrastive_infonce_loss, topology_deviation_ce
from mmsec_eval.attacks.advclip.patch import clamp_patch, patch_initialization, patch_tv, random_location
from mmsec_eval.attacks.advclip.gan_torch import DiscriminatorMLP, GeneratorMLP, Z_DIM
from mmsec_eval.config.schema import AppConfig
from mmsec_eval.datasets.registry import load_dataset
from mmsec_eval.plugins.registry import create
from mmsec_eval.runner.artifacts import (
    make_run_dir,
    new_run_id,
    write_env_snapshot,
    write_json_snapshot,
    write_results,
    write_summary,
)
from mmsec_eval.runner.report import write_report
from mmsec_eval.types import RunArtifacts
from mmsec_eval.utils.seed import set_seed

LOG = logging.getLogger(__name__)


# 中文注释：封装 _extract_clip_feats 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _extract_clip_feats(out: Any, *, attr_name: str):
    if hasattr(out, "detach") and hasattr(out, "shape"):
        return out
    v = getattr(out, attr_name, None)
    if v is not None:
        return v
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(out, (tuple, list)) and out:
        f = out[0]
        if hasattr(f, "detach") and hasattr(f, "shape"):
            return f
    raise TypeError(f"unsupported CLIP feature output type: {type(out)!r}")


# 中文注释：封装 _clip_adapter_ready 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _clip_adapter_ready(adapter: Any) -> bool:
    # Accept either:
    # 1) CLIP HF adapter (has _model/_processor), or
    # 2) any dual-stream adapter that exposes projected_features_torch.
    if hasattr(adapter, "projected_features_torch") and callable(getattr(adapter, "projected_features_torch")):
        return True
    return bool(getattr(adapter, "_ready", False) and getattr(adapter, "_model", None) is not None and getattr(adapter, "_processor", None) is not None)


# 中文注释：封装 _prepare_images_torch 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _prepare_images_torch(adapter: Any, images: list[np.ndarray]):
    """Convert a list of HWC float images in [0,1] to BCHW torch tensor on adapter.device."""
    import torch
    import torch.nn.functional as F

    device = getattr(adapter, "_device", "cpu")
    # CLIP image processor config if available; otherwise keep the original image size.
    target_hw: tuple[int, int] | None = None
    ip = getattr(getattr(adapter, "_processor", None), "image_processor", None)
    size = getattr(ip, "size", None) or {}
    if isinstance(size, dict) and "shortest_edge" in size:
        v = int(size["shortest_edge"])
        target_hw = (v, v)
    elif isinstance(size, dict) and "height" in size and "width" in size:
        target_hw = (int(size["height"]), int(size["width"]))

    xs = []
    for im in images:
        arr = np.asarray(im, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("image must be HWC with 3 channels")
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
        if target_hw is None:
            target_hw = (int(t.shape[2]), int(t.shape[3]))
        if int(t.shape[2]) != int(target_hw[0]) or int(t.shape[3]) != int(target_hw[1]):
            t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
        xs.append(t)
    x = torch.cat(xs, dim=0).to(device)
    x = x.clamp(0.0, 1.0)
    return x


# 中文注释：封装 _apply_patch_bchw 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _apply_patch_bchw(images, patch_chw, locs: list[tuple[int, int]]):
    """Apply a universal patch (CHW) to BCHW images (torch)."""
    import torch

    b, c, h, w = images.shape
    ph, pw = int(patch_chw.shape[-2]), int(patch_chw.shape[-1])
    out = images.clone()
    for i in range(int(b)):
        y, x = locs[i]
        y = int(max(0, min(h - 1, y)))
        x = int(max(0, min(w - 1, x)))
        y2 = int(min(h, y + ph))
        x2 = int(min(w, x + pw))
        ph2 = int(y2 - y)
        pw2 = int(x2 - x)
        if ph2 <= 0 or pw2 <= 0:
            continue
        out[i, :, y:y2, x:x2] = patch_chw[:, :ph2, :pw2]
    return out.clamp(0.0, 1.0)


# 中文注释：封装 _encode_clip_features 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _encode_clip_features(adapter: Any, images_bchw, texts: list[str]):
    """Return (image_features, text_features) as torch tensors (with gradients wrt images)."""
    import torch
    import torch.nn.functional as F

    if hasattr(adapter, "projected_features_torch") and callable(getattr(adapter, "projected_features_torch")):
        img_feat, txt_feat = adapter.projected_features_torch(images_bchw, texts)
        # Ensure float32 and normalized.
        img_feat = F.normalize(img_feat.float(), dim=-1)
        txt_feat = F.normalize(txt_feat.float(), dim=-1)
        return img_feat, txt_feat

    device = getattr(adapter, "_device", "cpu")
    model = getattr(adapter, "_model", None)
    processor = getattr(adapter, "_processor", None)
    if model is None or processor is None:
        raise RuntimeError("clip adapter missing model/processor")

    ip = getattr(processor, "image_processor", None)
    mean = getattr(ip, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(ip, "image_std", [0.26862954, 0.26130258, 0.27577711])
    mean_t = torch.tensor(mean, device=device, dtype=images_bchw.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=images_bchw.dtype).view(1, 3, 1, 1)
    pixel_values = (images_bchw - mean_t) / std_t

    tok = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    tok = {k: v.to(device) for k, v in tok.items()}

    img_raw = model.get_image_features(pixel_values=pixel_values)
    txt_raw = model.get_text_features(**tok)
    img_feat = _extract_clip_feats(img_raw, attr_name="image_embeds").float()
    txt_feat = _extract_clip_feats(txt_raw, attr_name="text_embeds").float()
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)
    return img_feat, txt_feat


# 中文注释：封装 _random_crop_patches 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _random_crop_patches(images_bchw, patch_size: int, rng: np.random.Generator, batch: int) -> Any:
    """Randomly crop real patches from BCHW images (torch tensor)."""
    import torch

    if images_bchw.ndim != 4:
        raise ValueError("images_bchw must be BCHW")
    b, c, h, w = images_bchw.shape
    p = int(patch_size)
    ph = min(p, int(h))
    pw = min(p, int(w))
    out: list[torch.Tensor] = []
    batch = int(min(max(1, batch), int(b)))
    for i in range(batch):
        y = int(rng.integers(0, max(1, int(h - ph + 1))))
        x = int(rng.integers(0, max(1, int(w - pw + 1))))
        out.append(images_bchw[i : i + 1, :, y : y + ph, x : x + pw])
    return torch.cat(out, dim=0)


# 中文注释：封装 _advclip_train_setup 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _advclip_train_setup(cfg: AppConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    run_id = new_run_id()
    run_dir = make_run_dir(cfg.artifacts_dir, run_id)
    attack_debug_root = Path(run_dir) / "attack_debug"
    attack_debug_root.mkdir(parents=True, exist_ok=True)

    write_json_snapshot(run_dir, "config_snapshot.json", asdict(cfg))
    write_env_snapshot(run_dir)

    surrogate_name = str(cfg.runner.surrogate_model_adapter or cfg.plugins.model_adapter)
    adapter = create("model_adapter", surrogate_name)
    mode = str(cfg.attack.mode).upper()
    patch_size = int(cfg.attack.patch_size)
    patch_path = str(attack_debug_root / f"advclip_patch_{mode}_{patch_size}.npy")
    reg_key = make_key(clip_model_name=str(cfg.model.clip_model_name), mode=mode, patch_size=patch_size)
    steps = int(cfg.attack.patch_train_steps or 0)
    if steps <= 0:
        steps = max(100, int(cfg.attack.epochs or 1) * 100)
    if not _clip_adapter_ready(adapter):
        raise RuntimeError(
            "AdvCLIP training requires a gradient-capable surrogate adapter exposing either "
            "(projected_features_torch) or (CLIP model+processor APIs). "
            f"Adapter={type(adapter).__name__} is not ready."
        )

    dataset = load_dataset(cfg)
    if cfg.runner.max_samples > 0:
        dataset = dataset[: cfg.runner.max_samples]
    if not dataset:
        raise ValueError("empty dataset")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "attack_debug_root": attack_debug_root,
        "surrogate_name": surrogate_name,
        "adapter": adapter,
        "mode": mode,
        "patch_size": patch_size,
        "patch_path": patch_path,
        "reg_key": reg_key,
        "steps": steps,
        "dataset": dataset,
        "batch_size": max(1, int(cfg.attack.batch_size or 8)),
        "lr": float(cfg.attack.step_size or 0.01),
        "device": getattr(adapter, "_device", "cpu"),
        "rng": np.random.default_rng(int(cfg.seed)),
        "use_gan": bool(cfg.attack.use_gan),
        "gan_steps": max(1, int(cfg.attack.gan_steps)),
        "gan_weight": float((cfg.attack.loss_weights or {}).get("gan", 1.0)),
    }


# 中文注释：封装 _advclip_batch 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _advclip_batch(ctx: dict[str, Any]) -> tuple[list[str], Any]:
    idx = ctx["rng"].integers(0, len(ctx["dataset"]), size=(ctx["batch_size"],))
    batch = [ctx["dataset"][int(i)] for i in idx.tolist()]
    texts = [str(s.text or "") for s in batch]
    images = [np.asarray(s.image, dtype=np.float32) for s in batch]
    images_t = _prepare_images_torch(ctx["adapter"], images).to(ctx["device"])
    return texts, images_t


# 中文注释：封装 _patch_regularizers 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _patch_regularizers(patch_chw: Any) -> tuple[Any, Any]:
    import torch

    tv = torch.abs(patch_chw[:, :, 1:] - patch_chw[:, :, :-1]).mean()
    tv = tv + torch.abs(patch_chw[:, 1:, :] - patch_chw[:, :-1, :]).mean()
    l2 = torch.mean((patch_chw - patch_chw.mean()) ** 2)
    return tv, l2


# 中文注释：封装 _advclip_objective 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _advclip_objective(cfg: AppConfig, adapter: Any, images_t: Any, texts: list[str], adv_images_t: Any) -> tuple[Any, Any, Any]:
    import torch

    with torch.no_grad():
        clean_img_feat, _ = _encode_clip_features(adapter, images_t, texts)
    adv_img_feat, txt_feat = _encode_clip_features(adapter, adv_images_t, texts)
    loss_align = contrastive_infonce_loss(adv_img_feat, txt_feat, tau=float(cfg.attack.tau_patch or 0.07))
    loss_tpd = topology_deviation_ce(
        clean_img_feat,
        adv_img_feat,
        topology_k=int(cfg.attack.topology_k),
        tau=float(cfg.attack.tau_patch or 0.07),
    )
    return loss_align, loss_tpd, txt_feat


# 中文注释：封装 _direct_patch_row 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _direct_patch_row(step: int, loss: Any, loss_align: Any, loss_tpd: Any, tv: Any, l2: Any) -> dict[str, float | int]:
    return {
        "step": int(step),
        "loss_total": float(loss.detach().cpu().item()),
        "loss_align": float(loss_align.detach().cpu().item()),
        "loss_tpd": float(loss_tpd.detach().cpu().item()),
        "loss_gan_d": 0.0,
        "loss_gan_g": 0.0,
        "d_real_mean": 0.0,
        "d_fake_mean": 0.0,
        "tv": float(tv.detach().cpu().item()),
        "l2": float(l2.detach().cpu().item()),
    }


# 中文注释：封装 _train_direct_patch 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _train_direct_patch(cfg: AppConfig, ctx: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    import torch

    patch_np = clamp_patch(patch_initialization(int(ctx["patch_size"]), seed=cfg.seed))
    patch = torch.from_numpy(patch_np).permute(2, 0, 1).to(ctx["device"])
    patch = patch.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([patch], lr=float(ctx["lr"]))
    rows: list[dict[str, Any]] = []
    for step in range(1, int(ctx["steps"]) + 1):
        texts, images_t = _advclip_batch(ctx)
        locs = [random_location((images_t.shape[2], images_t.shape[3], 3), int(ctx["patch_size"]), rng=ctx["rng"], margin=0) for _ in range(int(ctx["batch_size"]))]
        adv_images_t = _apply_patch_bchw(images_t, patch, locs)
        loss_align, loss_tpd, _ = _advclip_objective(cfg, ctx["adapter"], images_t, texts, adv_images_t)
        tv, l2 = _patch_regularizers(patch)
        loss = (
            -float(cfg.attack.lambda_at or 1.0) * loss_align
            -float(cfg.attack.lambda_tpd or 1.0) * loss_tpd
            + float(cfg.attack.tv_weight or 0.0) * tv
            + float(cfg.attack.nps_weight or 0.0) * l2
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)
        rows.append(_direct_patch_row(step, loss, loss_align, loss_tpd, tv, l2))
        _log_direct_patch_step(step, int(ctx["steps"]), rows[-1])
    patch_out = patch.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    np.save(str(ctx["patch_path"]), patch_out)
    return patch_out, rows, ""


# 中文注释：封装 _log_direct_patch_step 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _log_direct_patch_step(step: int, steps: int, row: dict[str, Any]) -> None:
    if step % 25 == 0 or step == steps:
        LOG.info(
            "advclip train step %d/%d loss=%.4f align=%.4f tpd=%.4f",
            step,
            steps,
            row["loss_total"],
            row["loss_align"],
            row["loss_tpd"],
        )


# 中文注释：封装 _gan_generators 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _gan_generators(device: Any, seed: int) -> tuple[Any, Any, Any]:
    import torch

    try:
        gen_device = torch.device(device).type
    except (RuntimeError, TypeError, ValueError):
        gen_device = "cuda" if str(device).startswith("cuda") else "cpu"
    torch_gen_train = torch.Generator(device=gen_device)
    torch_gen_train.manual_seed(int(seed) + 1)
    torch_gen_fixed = torch.Generator(device=gen_device)
    torch_gen_fixed.manual_seed(int(seed))
    z_fixed = torch.randn((1, Z_DIM), generator=torch_gen_fixed, device=device)
    return torch_gen_train, torch_gen_fixed, z_fixed


# 中文注释：封装 _train_discriminator 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _train_discriminator(ctx: dict[str, Any], gan: dict[str, Any], images_t: Any, step: int) -> dict[str, float]:
    import torch

    loss_d_total = 0.0
    d_real_total = 0.0
    d_fake_total = 0.0
    d_batch = int(min(int(ctx["batch_size"]), 8))
    for _ in range(int(ctx["gan_steps"])):
        real = _random_crop_patches(images_t, int(ctx["patch_size"]), rng=ctx["rng"], batch=d_batch)
        z = torch.randn((d_batch, Z_DIM), generator=gan["torch_gen_train"], device=ctx["device"])
        fake = gan["generator"](z)
        noise_sigma = max(0.0, 0.05 * (1.0 - float(step) / max(1.0, float(ctx["steps"]))))
        if noise_sigma > 0:
            real = (real + noise_sigma * torch.randn_like(real)).clamp(0.0, 1.0)
            fake = (fake + noise_sigma * torch.randn_like(fake)).clamp(0.0, 1.0)
        d_real = gan["discriminator"](real)
        d_fake = gan["discriminator"](fake.detach())
        loss_d = gan["bce"](d_real, torch.full_like(d_real, gan["real_label"]))
        loss_d = loss_d + gan["bce"](d_fake, torch.full_like(d_fake, gan["fake_label"]))
        gan["opt_d"].zero_grad(set_to_none=True)
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(gan["discriminator"].parameters(), max_norm=1.0)
        gan["opt_d"].step()
        loss_d_total += float(loss_d.detach().cpu().item())
        d_real_total += float(torch.sigmoid(d_real.detach()).mean().cpu().item())
        d_fake_total += float(torch.sigmoid(d_fake.detach()).mean().cpu().item())
    return {
        "loss_gan_d": float(loss_d_total / max(1, int(ctx["gan_steps"]))),
        "d_real_mean": float(d_real_total / max(1, int(ctx["gan_steps"]))),
        "d_fake_mean": float(d_fake_total / max(1, int(ctx["gan_steps"]))),
    }


# 中文注释：封装 _gan_generator_step 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _gan_generator_step(cfg: AppConfig, ctx: dict[str, Any], gan: dict[str, Any], texts: list[str], images_t: Any) -> dict[str, Any]:
    import torch

    z1 = torch.randn((1, Z_DIM), generator=gan["torch_gen_train"], device=ctx["device"])
    patch_chw = gan["generator"](z1)[0]
    locs = [random_location((images_t.shape[2], images_t.shape[3], 3), int(ctx["patch_size"]), rng=ctx["rng"], margin=0) for _ in range(int(ctx["batch_size"]))]
    adv_images_t = _apply_patch_bchw(images_t, patch_chw, locs)
    loss_align, loss_tpd, _ = _advclip_objective(cfg, ctx["adapter"], images_t, texts, adv_images_t)
    tv, l2 = _patch_regularizers(patch_chw)
    for p in gan["discriminator"].parameters():
        p.requires_grad_(False)
    d_patch = gan["discriminator"](patch_chw.unsqueeze(0))
    loss_gan_g_raw = gan["bce"](d_patch, torch.full_like(d_patch, gan["real_label"]))
    loss_gan_g = torch.clamp(loss_gan_g_raw, max=20.0)
    for p in gan["discriminator"].parameters():
        p.requires_grad_(True)
    loss = (
        -float(cfg.attack.lambda_at or 1.0) * loss_align
        -float(cfg.attack.lambda_tpd or 1.0) * loss_tpd
        + float(ctx["gan_weight"]) * loss_gan_g
        + float(cfg.attack.tv_weight or 0.0) * tv
        + float(cfg.attack.nps_weight or 0.0) * l2
    )
    gan["opt_g"].zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gan["generator"].parameters(), max_norm=1.0)
    gan["opt_g"].step()
    return {"loss": loss, "loss_align": loss_align, "loss_tpd": loss_tpd, "loss_gan_g": loss_gan_g, "loss_gan_g_raw": loss_gan_g_raw, "tv": tv, "l2": l2}


# 中文注释：封装 _gan_row 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _gan_row(step: int, generator_stats: dict[str, Any], discriminator_stats: dict[str, float]) -> dict[str, Any]:
    row = {
        "step": int(step),
        "loss_total": float(generator_stats["loss"].detach().cpu().item()),
        "loss_align": float(generator_stats["loss_align"].detach().cpu().item()),
        "loss_tpd": float(generator_stats["loss_tpd"].detach().cpu().item()),
        "loss_gan_g": float(generator_stats["loss_gan_g"].detach().cpu().item()),
        "loss_gan_g_raw": float(generator_stats["loss_gan_g_raw"].detach().cpu().item()),
        "tv": float(generator_stats["tv"].detach().cpu().item()),
        "l2": float(generator_stats["l2"].detach().cpu().item()),
    }
    row.update(discriminator_stats)
    return row


# 中文注释：封装 _init_gan_state 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _init_gan_state(ctx: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    import torch

    generator = GeneratorMLP(z_dim=Z_DIM, patch_size=int(ctx["patch_size"])).to(ctx["device"])
    discriminator = DiscriminatorMLP(patch_size=int(ctx["patch_size"])).to(ctx["device"])
    generator.train()
    discriminator.train()
    torch_gen_train, _, z_fixed = _gan_generators(ctx["device"], int(cfg.seed))
    return {
        "generator": generator,
        "discriminator": discriminator,
        "bce": torch.nn.BCEWithLogitsLoss(),
        "opt_g": torch.optim.Adam(generator.parameters(), lr=float(ctx["lr"]), betas=(0.5, 0.999)),
        "opt_d": torch.optim.Adam(discriminator.parameters(), lr=float(ctx["lr"]) * 0.5, betas=(0.5, 0.999)),
        "real_label": 0.9,
        "fake_label": 0.1,
        "torch_gen_train": torch_gen_train,
        "z_fixed": z_fixed,
    }


# 中文注释：封装 _train_gan_patch 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _train_gan_patch(cfg: AppConfig, ctx: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    import torch

    gan = _init_gan_state(ctx, cfg)
    rows: list[dict[str, Any]] = []
    for step in range(1, int(ctx["steps"]) + 1):
        texts, images_t = _advclip_batch(ctx)
        d_stats = _train_discriminator(ctx, gan, images_t, step)
        g_stats = _gan_generator_step(cfg, ctx, gan, texts, images_t)
        rows.append(_gan_row(step, g_stats, d_stats))
        _log_gan_step(step, int(ctx["steps"]), rows[-1])
    patch_out_t = gan["generator"](gan["z_fixed"])[0].detach().cpu()
    patch_out = patch_out_t.permute(1, 2, 0).numpy().astype(np.float32)
    np.save(str(ctx["patch_path"]), patch_out)
    gan_state_path = str(ctx["attack_debug_root"] / "advclip_gan_state.pt")
    torch.save(_gan_state_payload(ctx, cfg, gan), gan_state_path)
    return patch_out, rows, gan_state_path


# 中文注释：封装 _log_gan_step 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _log_gan_step(step: int, steps: int, row: dict[str, Any]) -> None:
    if step % 25 == 0 or step == steps:
        LOG.info(
            "advclip gan step %d/%d loss=%.4f align=%.4f tpd=%.4f gan_g=%.4f gan_d=%.4f",
            step,
            steps,
            row["loss_total"],
            row["loss_align"],
            row["loss_tpd"],
            row["loss_gan_g"],
            row["loss_gan_d"],
        )


# 中文注释：封装 _gan_state_payload 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _gan_state_payload(ctx: dict[str, Any], cfg: AppConfig, gan: dict[str, Any]) -> dict[str, Any]:
    return {
        "generator": gan["generator"].state_dict(),
        "discriminator": gan["discriminator"].state_dict(),
        "z_dim": int(Z_DIM),
        "patch_size": int(ctx["patch_size"]),
        "mode": str(ctx["mode"]),
        "run_id": str(ctx["run_id"]),
        "clip_model_name": str(cfg.model.clip_model_name),
    }


# 中文注释：封装 _save_patch_preview 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _save_patch_preview(patch_out: np.ndarray, attack_debug_root: Path) -> str:
    try:
        preview = (np.clip(patch_out, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(preview).save(attack_debug_root / "advclip_patch_preview.png")
        return ""
    except (OSError, ValueError) as exc:
        return str(exc)


# 中文注释：封装 _finalize_advclip_training 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _finalize_advclip_training(cfg: AppConfig, ctx: dict[str, Any], patch_out: np.ndarray, rows: list[dict[str, Any]], gan_state_path: str) -> RunArtifacts:
    preview_error = _save_patch_preview(patch_out, ctx["attack_debug_root"])
    summary = _advclip_summary(cfg, ctx, patch_out, rows, gan_state_path, preview_error)
    update_entry(
        artifacts_dir=str(cfg.artifacts_dir),
        key=str(ctx["reg_key"]),
        patch_path=str(ctx["patch_path"]),
        run_id=str(ctx["run_id"]),
        trained=True,
        use_gan=bool(ctx["use_gan"]),
    )
    results_path = write_results(str(ctx["run_dir"]), rows)
    summary_path = write_summary(str(ctx["run_dir"]), summary)
    write_json_snapshot(
        str(ctx["run_dir"]),
        "report_data.json",
        {
            "summary": summary,
            "rows_preview": rows[: min(50, len(rows))],
            "metric_series": {"loss": [float(r["loss_total"]) for r in rows]},
            "reproduction_fidelity": [{"paper": "AdvCLIP", "status": "approx", "source": "src/mmsec_eval/attacks/advclip/*"}],
        },
    )
    report_path = write_report(str(ctx["run_dir"]), summary=summary, rows=rows[-50:])
    return RunArtifacts(
        run_id=str(ctx["run_id"]),
        run_dir=str(ctx["run_dir"]),
        results_path=results_path,
        summary_path=summary_path,
        report_path=report_path,
        run_index_path="",
        benchmark_summary_path="",
    )


# 中文注释：封装 _advclip_summary 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
def _advclip_summary(cfg: AppConfig, ctx: dict[str, Any], patch_out: np.ndarray, rows: list[dict[str, Any]], gan_state_path: str, preview_error: str) -> dict[str, Any]:
    return {
        "run_id": str(ctx["run_id"]),
        "task_kind": "advclip_train",
        "trained": True,
        "surrogate_model_adapter": str(ctx["surrogate_name"]),
        "patch_path": str(ctx["patch_path"]),
        "patch_size": int(ctx["patch_size"]),
        "mode": str(ctx["mode"]),
        "use_gan": bool(ctx["use_gan"]),
        "gan_steps": int(ctx["gan_steps"]),
        "gan_weight": float(ctx["gan_weight"]),
        "gan_state_path": gan_state_path,
        "steps": int(ctx["steps"]),
        "final": rows[-1] if rows else {},
        "patch_mean": float(patch_out.mean()),
        "patch_std": float(patch_out.std()),
        "tv": float(patch_tv(patch_out)),
        "preview_error": preview_error,
    }


# 中文注释：实现 train_advclip_patch 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def train_advclip_patch(cfg: AppConfig) -> RunArtifacts:
    """Train and save an AdvCLIP universal patch into a new run directory."""
    ctx = _advclip_train_setup(cfg)
    if bool(ctx["use_gan"]):
        patch_out, rows, gan_state_path = _train_gan_patch(cfg, ctx)
    else:
        patch_out, rows, gan_state_path = _train_direct_patch(cfg, ctx)
    return _finalize_advclip_training(cfg, ctx, patch_out, rows, gan_state_path)
