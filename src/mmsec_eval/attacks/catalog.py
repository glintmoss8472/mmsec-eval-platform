# 文件说明：该文件属于攻击算法公共层，集中实现 catalog 相关逻辑。
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


LOCAL_TORCH_SURROGATE_ADAPTERS: FrozenSet[str] = frozenset({"clip_hf", "blip_itm", "vilt_itm"})
CLIP_ONLY_SURROGATE_ADAPTERS: FrozenSet[str] = frozenset({"clip_hf"})

CLASSIC_GRADIENT_ATTACKS: FrozenSet[str] = frozenset(
    {"fgsm", "bim", "pgd", "mifgsm", "nifgsm", "difgsm", "tifgsm", "dtmifgsm", "vmifgsm", "vnifgsm", "cw"}
)
JOINT_TEXT_ATTACKS: FrozenSet[str] = frozenset({"tmm", "advedm_plus"})
ADVEDM_FAMILY_ATTACKS: FrozenSet[str] = frozenset({"advedm", "advedm_plus"})
INTERNAL_CORRUPTION_ATTACKS: FrozenSet[str] = frozenset()
EXTERNAL_REPO_ATTACKS: FrozenSet[str] = frozenset({"vqa_visual_corruption", "foa_attack", "anyattack", "mpc_attack", "m_attack"})
EXTERNAL_PACKAGE_ATTACKS: FrozenSet[str] = frozenset({"xtransfer_uap"})
EXTERNAL_ATTACKS: FrozenSet[str] = EXTERNAL_REPO_ATTACKS | EXTERNAL_PACKAGE_ATTACKS
PAPER_ATTACKS: FrozenSet[str] = frozenset(
    {"advclip", "tmm", "advedm", "advedm_plus", "vqa_visual_corruption", "xtransfer_uap", "foa_attack", "anyattack", "mpc_attack", "m_attack"}
)


# 定义 `AttackSurrogatePolicy` 的状态和行为边界，供攻击算法公共层在固定职责内复用。
@dataclass(frozen=True)
class AttackSurrogatePolicy:
    attacks: FrozenSet[str]
    supported_surrogates: FrozenSet[str]
    message: str


# 定义 `ExternalAttackMetadata` 的状态和行为边界，供攻击算法公共层在固定职责内复用。
@dataclass(frozen=True)
class ExternalAttackMetadata:
    attack_id: str
    display_name: str
    attack_mode: str
    attack_scope: str
    requires_external_repo: bool
    requires_checkpoint: bool
    requires_target_image: bool
    supports_blackbox_victim: bool
    supports_video: bool
    training: bool
    recommended_tasks: tuple[str, ...]


EXTERNAL_ATTACK_METADATA: dict[str, ExternalAttackMetadata] = {
    "vqa_visual_corruption": ExternalAttackMetadata("vqa_visual_corruption", "官方视觉退化攻击（VQA Visual Robustness）", "official_visual_corruption", "image", True, False, False, True, False, False, ("vqa", "caption", "vlr")),
    "xtransfer_uap": ExternalAttackMetadata("xtransfer_uap", "跨任务通用扰动（X-Transfer UAP）", "universal_perturbation", "image", False, True, False, True, False, False, ("vlr", "caption", "vqa")),
    "foa_attack": ExternalAttackMetadata("foa_attack", "特征最优对齐迁移攻击（FOA-Attack）", "targeted_transfer", "image", True, False, True, True, False, False, ("caption", "vqa", "vlr")),
    "anyattack": ExternalAttackMetadata("anyattack", "任意图像目标生成攻击（AnyAttack）", "pretrained_generator_targeted", "image", True, True, True, True, False, False, ("caption", "vqa", "vlr")),
    "mpc_attack": ExternalAttackMetadata("mpc_attack", "多范式协同迁移攻击（MPCAttack）", "multi_paradigm_collaborative_transfer", "image", True, False, True, True, False, False, ("caption", "vqa", "vlr")),
    "m_attack": ExternalAttackMetadata("m_attack", "局部语义匹配迁移攻击（M-Attack）", "targeted_transfer_local_semantic", "image", True, False, True, True, False, False, ("caption", "vqa", "vlr")),
}


SURROGATE_POLICIES: tuple[AttackSurrogatePolicy, ...] = (
    AttackSurrogatePolicy(
        attacks=frozenset({"tmm"}),
        supported_surrogates=LOCAL_TORCH_SURROGATE_ADAPTERS,
        message=(
            "TMM 只支持具备 attention_map、score_pairs_torch 和 projected_features_torch 的本地代理模型："
            "clip_hf、blip_itm、vilt_itm。当前选择的模型只能作为受测模型。"
        ),
    ),
    AttackSurrogatePolicy(
        attacks=ADVEDM_FAMILY_ATTACKS,
        supported_surrogates=CLIP_ONLY_SURROGATE_ADAPTERS,
        message=(
            "AdvEDM / AdvEDM+ 当前只支持 clip_hf 作为代理模型，因为图像优化链路需要 patch_text_similarity_torch、"
            "attention_map 和 Torch 侧梯度打分。当前选择的模型只能作为受测模型。"
        ),
    ),
    AttackSurrogatePolicy(
        attacks=CLASSIC_GRADIENT_ATTACKS,
        supported_surrogates=LOCAL_TORCH_SURROGATE_ADAPTERS,
        message=(
            "经典梯度攻击当前只支持具备 score_pairs_torch 的本地代理模型："
            "clip_hf、blip_itm、vilt_itm。OpenAI 兼容视觉-语言受测模型当前只能作为受测模型。"
        ),
    ),
)


# 推断 `surrogate policy 所属 攻击`，从样本、配置或运行记录中提取统一名称。
def surrogate_policy_for_attack(attack: str) -> AttackSurrogatePolicy | None:
    attack_id = str(attack or "").strip()
    for policy in SURROGATE_POLICIES:
        if attack_id in policy.attacks:
            return policy
    return None


# 推断 `攻击 surrogate error`，从样本、配置或运行记录中提取统一名称。
def attack_surrogate_error(attack: str, surrogate: str) -> str | None:
    attack_id = str(attack or "").strip()
    surrogate_id = str(surrogate or "").strip()
    if not attack_id or not surrogate_id:
        return None
    policy = surrogate_policy_for_attack(attack_id)
    if policy is None:
        return None
    if surrogate_id in policy.supported_surrogates:
        return None
    return policy.message
