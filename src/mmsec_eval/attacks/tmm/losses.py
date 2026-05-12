# 文件说明：该文件属于TMM 迁移攻击模块，集中实现 losses 相关逻辑。
from __future__ import annotations

import torch
import torch.nn.functional as F


# 中文注释：实现 cosine_similarity 的核心流程，支撑TMM 迁移攻击模块中的业务语义和异常边界。
def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.float().reshape(a.shape[0], -1) if a.ndim > 1 else a.float().reshape(1, -1)
    b = b.float().reshape(b.shape[0], -1) if b.ndim > 1 else b.float().reshape(1, -1)
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


# 中文注释：实现 ssim_loss 的核心流程，支撑TMM 迁移攻击模块中的业务语义和异常边界。
def ssim_loss(x: torch.Tensor, y: torch.Tensor, window: int = 11, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable SSIM-like loss: 1 - SSIM (lower is better)."""
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("x/y must be BCHW")
    if x.shape != y.shape:
        raise ValueError("x/y shape mismatch")

    # Convert to grayscale to keep it cheap and stable.
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    xg = (x * w).sum(dim=1, keepdim=True)
    yg = (y * w).sum(dim=1, keepdim=True)

    # Use average pooling as a simple window.
    pad = window // 2
    mu_x = F.avg_pool2d(xg, kernel_size=window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(yg, kernel_size=window, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(xg * xg, kernel_size=window, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(yg * yg, kernel_size=window, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(xg * yg, kernel_size=window, stride=1, padding=pad) - mu_x * mu_y

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = num / (den + eps)
    return 1.0 - ssim.mean()


# 中文注释：实现 ogfh_lo 的核心流程，支撑TMM 迁移攻击模块中的业务语义和异常边界。
def ogfh_lo(
    *,
    img_feat_adv: torch.Tensor,
    img_feat_clean: torch.Tensor,
    txt_feat: torch.Tensor,
    txt_feat_clean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Orthogonal-guided feature heterogenization (OGFH).

    Eq.(5)~(7) spirit:
      - Lot: orthogonality between eo and et
      - Lov: orthogonality between eo and ev
      - Loa: orthogonality between eo and ea
      - LO = Loa + Lot + Lov
    """
    img_adv = F.normalize(img_feat_adv.float(), dim=-1)
    img_clean = F.normalize(img_feat_clean.float(), dim=-1)
    txt_adv = F.normalize(txt_feat.float(), dim=-1)
    txt_clean = F.normalize((txt_feat_clean if txt_feat_clean is not None else txt_feat).float(), dim=-1)

    # Fused embeddings for eo, et, ev, ea.
    eo = F.normalize(torch.cat([img_clean, txt_clean], dim=-1), dim=-1)
    et = F.normalize(torch.cat([img_clean, txt_adv], dim=-1), dim=-1)
    ev = F.normalize(torch.cat([img_adv, txt_clean], dim=-1), dim=-1)
    ea = F.normalize(torch.cat([img_adv, txt_adv], dim=-1), dim=-1)

    lot = ((eo * et).sum(dim=-1) ** 2).mean()
    lov = ((eo * ev).sum(dim=-1) ** 2).mean()
    loa = ((eo * ea).sum(dim=-1) ** 2).mean()
    lo = loa + lot + lov
    return lo, {"Lov": lov, "Lot": lot, "Loa": loa}
