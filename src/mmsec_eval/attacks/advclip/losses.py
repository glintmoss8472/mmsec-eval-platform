# 文件说明：该文件属于AdvCLIP 攻击模块，集中实现 losses 相关逻辑。
from __future__ import annotations

import torch
import torch.nn.functional as F


# 中文注释：实现 contrastive_infonce_loss 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def contrastive_infonce_loss(image_features: torch.Tensor, text_features: torch.Tensor, *, tau: float = 0.07) -> torch.Tensor:
    """CLIP-style symmetric InfoNCE loss (lower is better alignment).

    image_features: [B, D]
    text_features:  [B, D]
    """
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("features must be 2D")
    if image_features.shape != text_features.shape:
        raise ValueError(f"feature shape mismatch: img={image_features.shape} txt={text_features.shape}")

    img = F.normalize(image_features.float(), dim=-1)
    txt = F.normalize(text_features.float(), dim=-1)
    logits = (img @ txt.t()) / max(float(tau), 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)


# 中文注释：实现 topology_deviation_ce 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def topology_deviation_ce(
    clean_features: torch.Tensor,
    adv_features: torch.Tensor,
    *,
    topology_k: int = 5,
    tau: float = 0.07,
) -> torch.Tensor:
    """Topology deviation loss on clean/adv neighborhood graphs.

    We build row-wise neighborhood probability distributions from clean/adv similarity
    matrices, then measure divergence (clean -> adv). A larger value means stronger
    topology corruption, matching AdvCLIP's objective design.
    """
    if clean_features.ndim != 2 or adv_features.ndim != 2:
        raise ValueError("features must be 2D")
    if clean_features.shape != adv_features.shape:
        raise ValueError(f"feature shape mismatch: clean={clean_features.shape} adv={adv_features.shape}")
    b = int(clean_features.shape[0])
    if b <= 1:
        return torch.zeros((), device=clean_features.device, dtype=torch.float32)

    clean = F.normalize(clean_features.float(), dim=-1)
    adv = F.normalize(adv_features.float(), dim=-1)
    eps = 1e-8

    clean_sim = (clean @ clean.t()) / max(float(tau), 1e-6)
    adv_sim = (adv @ adv.t()) / max(float(tau), 1e-6)

    # Exclude self from neighborhood graph.
    eye = torch.eye(b, device=clean_sim.device, dtype=torch.bool)
    clean_sim = clean_sim.masked_fill(eye, float("-inf"))
    adv_sim = adv_sim.masked_fill(eye, float("-inf"))

    # Keep only top-k neighbors from clean graph (paper-aligned configurable locality).
    k = int(max(1, topology_k))
    k = min(k, max(1, b - 1))
    knn_idx = torch.topk(clean_sim, k=k, dim=1).indices
    knn_mask = torch.zeros_like(clean_sim, dtype=torch.bool)
    knn_mask.scatter_(1, knn_idx, True)

    clean_prob = torch.softmax(clean_sim.masked_fill(~knn_mask, float("-inf")), dim=1)
    adv_prob = torch.softmax(adv_sim.masked_fill(~knn_mask, float("-inf")), dim=1)

    # CE-like divergence (same spirit as official implementation's graph divergence).
    kl_main = (clean_prob * ((clean_prob + eps).log() - (adv_prob + eps).log())).sum(dim=1)
    clean_comp = (1.0 - clean_prob).clamp_min(eps)
    adv_comp = (1.0 - adv_prob).clamp_min(eps)
    kl_comp = (clean_comp * (clean_comp.log() - adv_comp.log())).sum(dim=1)
    return (kl_main + kl_comp).mean()


# 中文注释：实现 reconstruction_mse 的核心流程，支撑AdvCLIP 攻击模块中的业务语义和异常边界。
def reconstruction_mse(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(adv.float(), clean.float())
