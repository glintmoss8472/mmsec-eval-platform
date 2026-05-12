# 文件说明：该文件属于自动化测试，集中实现 test advclip losses 相关逻辑。
from __future__ import annotations

import torch

from mmsec_eval.attacks.advclip.losses import topology_deviation_ce


# 中文注释：验证 test_topology_deviation_small_when_clean_equals_adv 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_topology_deviation_small_when_clean_equals_adv():
    torch.manual_seed(0)
    clean = torch.randn(16, 32, dtype=torch.float32)
    same = clean.clone()
    shuffled = clean[torch.randperm(clean.shape[0])]

    l_same = topology_deviation_ce(clean, same, topology_k=5, tau=0.07)
    l_shuf = topology_deviation_ce(clean, shuffled, topology_k=5, tau=0.07)

    assert float(l_same.item()) < float(l_shuf.item())


# 中文注释：验证 test_topology_deviation_supports_different_k 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_topology_deviation_supports_different_k():
    torch.manual_seed(1)
    clean = torch.randn(10, 24, dtype=torch.float32)
    adv = torch.randn(10, 24, dtype=torch.float32)
    l1 = topology_deviation_ce(clean, adv, topology_k=1, tau=0.07)
    l5 = topology_deviation_ce(clean, adv, topology_k=5, tau=0.07)
    assert torch.isfinite(l1)
    assert torch.isfinite(l5)
