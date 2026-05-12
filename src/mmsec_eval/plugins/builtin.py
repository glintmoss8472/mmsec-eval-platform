from __future__ import annotations

from mmsec_eval.attacks.advedm.attack import ADVEDMAttack
from mmsec_eval.attacks.advedm_plus.attack import ADVEDMPlusAttack
from mmsec_eval.attacks.advclip.attack import AdvCLIPPatchAttack
from mmsec_eval.attacks.classic_gradient import (
    BIMAttack,
    CWAttack,
    DIFGSMAttack,
    DTMIFGSMAttack,
    FGSMAttack,
    MIFGSMAttack,
    NIFGSMAttack,
    PGDAttack,
    TIFGSMAttack,
    VMIFGSMAttack,
    VNIFGSMAttack,
)
from mmsec_eval.attacks.external import (
    AnyAttackPlugin,
    FOAAttack,
    MPCAttackPlugin,
    MAttackPlugin,
    VQAVisualCorruptionAttack,
    XTransferUAPAttack,
)
from mmsec_eval.attacks.tmm.attack import TMMAttack
from mmsec_eval.defenses.sanitize_v1 import SanitizeDefenseV1
from mmsec_eval.judges.llm_judge import LLMJudge
from mmsec_eval.judges.rule_judge import RuleJudge
from mmsec_eval.metrics.basic_metrics import BasicMetrics
from mmsec_eval.metrics.ssim_metric import SSIMMetric
from mmsec_eval.model_adapters.blip_itm_adapter import BlipITMAdapter
from mmsec_eval.model_adapters.clip_hf_adapter import ClipHFAdapter
from mmsec_eval.model_adapters.gemini_adapter import GeminiVisionAdapter
from mmsec_eval.model_adapters.fixture_vlm_adapter import FixtureVLMAdapter
from mmsec_eval.model_adapters.http_adapter import HttpAdapter
from mmsec_eval.model_adapters.local_vlm_catalog import LOCAL_OPENAI_COMPAT_MODEL_SPECS
from mmsec_eval.model_adapters.openai_compat_adapter import OpenAICompatAdapter
from mmsec_eval.model_adapters.vilt_itm_adapter import ViltITMAdapter
from mmsec_eval.plugins.registry import register


def _register_openai_compatible_adapters() -> None:
    for spec in LOCAL_OPENAI_COMPAT_MODEL_SPECS:
        register(
            "model_adapter",
            spec.adapter,
            lambda variant=spec.variant: OpenAICompatAdapter(variant=variant),
        )


def register_builtin_plugins() -> None:
    register("model_adapter", "clip_hf", lambda: ClipHFAdapter())
    register("model_adapter", "blip_itm", lambda: BlipITMAdapter())
    register("model_adapter", "vilt_itm", lambda: ViltITMAdapter())
    register("model_adapter", "http", lambda: HttpAdapter())
    register("model_adapter", "openai_compat", lambda: OpenAICompatAdapter())
    register("model_adapter", "openai_gpt4o", lambda: OpenAICompatAdapter(variant="GPT4O"))
    _register_openai_compatible_adapters()
    register("model_adapter", "gemini_vision", lambda: GeminiVisionAdapter())
    register("model_adapter", "fixture_vlm", lambda: FixtureVLMAdapter())

    register("attack", "advedm", lambda: ADVEDMAttack())
    register("attack", "advedm_plus", lambda: ADVEDMPlusAttack())
    register("attack", "advclip", lambda: AdvCLIPPatchAttack())
    register("attack", "tmm", lambda: TMMAttack())
    register("attack", "fgsm", lambda: FGSMAttack())
    register("attack", "bim", lambda: BIMAttack())
    register("attack", "pgd", lambda: PGDAttack())
    register("attack", "mifgsm", lambda: MIFGSMAttack())
    register("attack", "nifgsm", lambda: NIFGSMAttack())
    register("attack", "difgsm", lambda: DIFGSMAttack())
    register("attack", "tifgsm", lambda: TIFGSMAttack())
    register("attack", "dtmifgsm", lambda: DTMIFGSMAttack())
    register("attack", "vmifgsm", lambda: VMIFGSMAttack())
    register("attack", "vnifgsm", lambda: VNIFGSMAttack())
    register("attack", "cw", lambda: CWAttack())
    register("attack", "vqa_visual_corruption", lambda: VQAVisualCorruptionAttack())
    register("attack", "xtransfer_uap", lambda: XTransferUAPAttack())
    register("attack", "foa_attack", lambda: FOAAttack())
    register("attack", "anyattack", lambda: AnyAttackPlugin())
    register("attack", "mpc_attack", lambda: MPCAttackPlugin())
    register("attack", "m_attack", lambda: MAttackPlugin())
    register("defense", "sanitize_v1", lambda: SanitizeDefenseV1())

    register("metric", "basic", lambda: BasicMetrics())
    register("metric", "ssim", lambda: SSIMMetric())

    register("judge", "rule", lambda: RuleJudge())
    register("judge", "llm", lambda: LLMJudge())
