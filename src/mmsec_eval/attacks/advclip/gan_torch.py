# 文件说明：该文件属于AdvCLIP 攻击模块，集中实现 gan torch 相关逻辑。
from __future__ import annotations

import torch
from torch import nn


Z_DIM = 128


# 实现 `GeneratorMLP.__init__` 的对象行为，维护该类在AdvCLIP 攻击模块中的调用契约。
class GeneratorMLP(nn.Module):
    # 封装 GeneratorMLP.__init__ 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
    def __init__(self, *, z_dim: int = Z_DIM, patch_size: int) -> None:
        super().__init__()
        p = int(patch_size)
        self.z_dim = int(z_dim)
        self.patch_size = p
        out_dim = 3 * p * p
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.Sigmoid(),  # output in [0,1]
        )

    # 实现 `GeneratorMLP.forward` 的对象行为，维护该类在AdvCLIP 攻击模块中的调用契约。
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z.float())
        b = int(x.shape[0])
        p = int(self.patch_size)
        return x.view(b, 3, p, p)


# 实现 `DiscriminatorMLP.__init__` 的对象行为，维护该类在AdvCLIP 攻击模块中的调用契约。
class DiscriminatorMLP(nn.Module):
    # 封装 DiscriminatorMLP.__init__ 的内部步骤，让AdvCLIP 攻击模块主流程保持清晰并隔离边界细节。
    def __init__(self, *, patch_size: int) -> None:
        super().__init__()
        p = int(patch_size)
        in_dim = 3 * p * p
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    # 实现 `DiscriminatorMLP.forward` 的对象行为，维护该类在AdvCLIP 攻击模块中的调用契约。
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        # patch: [B,3,P,P] in [0,1]
        if patch.ndim != 4:
            raise ValueError("patch must be Bx3xPxP")
        x = patch.reshape(patch.shape[0], -1).float()
        return self.net(x).squeeze(-1)  # [B]

