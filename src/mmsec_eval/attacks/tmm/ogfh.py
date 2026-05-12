# 文件说明：该文件属于TMM 迁移攻击模块，集中实现 ogfh 相关逻辑。
from __future__ import annotations

import torch


# 执行 `cosine similarity` 辅助逻辑，保持TMM 迁移攻击模块中的输入处理和结果输出一致。
def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.reshape(a.shape[0], -1).float()
    b = b.reshape(b.shape[0], -1).float()
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


# 执行 `orthogonalize grad` 辅助逻辑，保持TMM 迁移攻击模块中的输入处理和结果输出一致。
def orthogonalize_grad(grad: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Remove the projection of grad onto ref (per sample)."""
    g = grad.reshape(grad.shape[0], -1).float()
    r = ref.reshape(ref.shape[0], -1).float()
    denom = (r * r).sum(dim=1, keepdim=True) + eps
    coef = (g * r).sum(dim=1, keepdim=True) / denom
    g_orth = g - coef * r
    return g_orth.reshape_as(grad)

