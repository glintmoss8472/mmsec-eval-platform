# 文件说明：该文件属于模型适配层，集中实现 vilt itm adapter 相关逻辑。
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from mmsec_eval.model_adapters.image_utils import bchw_to_pil_images, processor_target_hw, stack_resized_rgb01
from mmsec_eval.plugins.base import ModelAdapter
from mmsec_eval.model_adapters.hf_local import hf_load_failure_message, require_cuda_device, resolve_hf_model_source
from mmsec_eval.types import ModelOutput, Sample

LOG = logging.getLogger(__name__)


# 执行 `pick 文本 length limit` 辅助逻辑，保持模型适配层中的输入处理和结果输出一致。
def _pick_text_length_limit(*values: Any) -> int:
    valid: list[int] = []
    for value in values:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            continue
        if ivalue <= 0:
            continue
        # Ignore unbounded sentinel values such as 1e30.
        if ivalue > 4096:
            continue
        valid.append(ivalue)
    return min(valid) if valid else 40


# 定义 `ViltITMAdapter` 的插件适配边界，把模型、攻击或评测能力暴露为统一接口。
class ViltITMAdapter(ModelAdapter):
    """ViLT retrieval (cross-encoder) adapter, CUDA-only and strict."""

    # 实现 `ViltITMAdapter.__init__` 的对象行为，维护该类在模型适配层中的调用契约。
    def __init__(self) -> None:
        import torch
        from transformers import ViltForImageAndTextRetrieval, ViltProcessor

        self._device = require_cuda_device("ViltITMAdapter", torch)
        self._model_name = os.getenv("MMSEC_VILT_ITM_MODEL_NAME", "dandelin/vilt-b32-finetuned-coco")
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        local_only = os.getenv("MMSEC_HF_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}
        source = resolve_hf_model_source(self._model_name, local_only=local_only, local_dir_name="vilt_itm")
        LOG.info("loading ViLT model=%s source=%s device=%s", self._model_name, source, self._device)
        try:
            self._model = ViltForImageAndTextRetrieval.from_pretrained(
                source, local_files_only=local_only
            ).to(self._device)
            self._model.eval()
            self._processor = ViltProcessor.from_pretrained(source, local_files_only=local_only)
            tokenizer = getattr(self._processor, "tokenizer", None)
            self._max_text_length = _pick_text_length_limit(
                getattr(tokenizer, "model_max_length", None),
                getattr(getattr(self._model, "config", None), "max_position_embeddings", None),
                getattr(getattr(getattr(self._model, "config", None), "text_config", None), "max_position_embeddings", None),
            )
        except (OSError, RuntimeError, ValueError) as e:
            raise RuntimeError(
                hf_load_failure_message(
                    adapter_label="ViLT",
                    model_name=self._model_name,
                    source=source,
                    device=self._device,
                    local_only=local_only,
                    cause=e,
                )
            ) from e

    # 实现 `ViltITMAdapter.device` 的对象行为，维护该类在模型适配层中的调用契约。
    @property
    def device(self) -> str:
        return str(self._device)

    # 准备 `inputs PyTorch` 数据，补齐后续运行、报告或测试需要的字段。
    def _prepare_inputs_torch(self, images_bchw, texts: list[str]):
        if images_bchw.ndim != 4:
            raise ValueError("images must be BCHW")
        pil = bchw_to_pil_images(images_bchw)
        enc = self._processor(
            images=pil,
            text=[str(text) for text in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self._max_text_length),
        )
        out: dict[str, Any] = {}
        for k, v in enc.items():
            out[k] = v.to(self._device)
        return out

    # 准备 `numpy 批处理` 数据，补齐后续运行、报告或测试需要的字段。
    def _prepare_numpy_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return stack_resized_rgb01(images, size_hw=processor_target_hw(self._processor, default_hw=(384, 384)))

    # 计算 `pairs PyTorch`，为指标、风险或调度决策提供数值依据。
    def score_pairs_torch(self, images_bchw, texts: list[str], *, output_attentions: bool = False):
        inp = self._prepare_inputs_torch(images_bchw, texts)
        out = self._model(**inp, output_attentions=bool(output_attentions), return_dict=True)
        return out.logits.squeeze(-1).float()

    # 实现 `ViltITMAdapter.projected_features_torch` 的对象行为，维护该类在模型适配层中的调用契约。
    def projected_features_torch(self, images_bchw, texts: list[str]):
        inp = self._prepare_inputs_torch(images_bchw, texts)
        out = self._model.vilt(**inp, output_attentions=False, output_hidden_states=False, return_dict=True)
        pooled = out.pooler_output.float()
        # ViLT retrieval is cross-encoder; for compatibility we expose same vector twice.
        return pooled, pooled

    # 计算 `pairs`，为指标、风险或调度决策提供数值依据。
    def score_pairs(self, pairs: list[tuple[np.ndarray, str]], batch_size: int = 8) -> np.ndarray:
        import torch

        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        out: list[np.ndarray] = []
        step = max(1, int(batch_size))
        for i in range(0, len(pairs), step):
            chunk = pairs[i : i + step]
            x = self._prepare_numpy_batch([im for im, _ in chunk])
            texts = [str(t) for _, t in chunk]
            x_t = torch.from_numpy(x).permute(0, 3, 1, 2).to(self._device)
            with torch.no_grad():
                scores = self.score_pairs_torch(x_t, texts)
            out.append(scores.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0).astype(np.float32)

    # 实现 `ViltITMAdapter.fused_embedding` 的对象行为，维护该类在模型适配层中的调用契约。
    def fused_embedding(self, image: np.ndarray, text: str) -> np.ndarray:
        import torch

        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pooled, _ = self.projected_features_torch(x, [str(text)])
        return pooled[0].detach().cpu().numpy().astype(np.float32)

    # 实现 `ViltITMAdapter.attention_map` 的对象行为，维护该类在模型适配层中的调用契约。
    def attention_map(self, image: np.ndarray, text: str, eps: float = 1e-8) -> np.ndarray:
        import math
        import torch

        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(self._device)
        inp = self._prepare_inputs_torch(x, [str(text)])
        with torch.no_grad():
            out = self._model(**inp, output_attentions=True, return_dict=True)
            attns = out.attentions or []
            if not attns:
                raise RuntimeError("ViLT attentions are unavailable")
            last = attns[-1].float()
            n_text = int(inp["input_ids"].shape[1])
            cls_to = last.mean(dim=1)[:, 0, 1 + n_text :]
            vec = cls_to[0].detach().cpu().numpy().astype(np.float32)

        n = int(vec.shape[0])
        side = int(math.sqrt(n))
        if side * side == n:
            m = vec.reshape(side, side)
        else:
            m = vec.reshape(1, -1)
        m = m - float(m.min())
        m = m / float(m.max() + eps)

        h, w = int(image.shape[0]), int(image.shape[1])
        if m.shape != (h, w):
            import torch.nn.functional as F

            t = torch.from_numpy(m[None, None, ...])
            t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
            m = t[0, 0].numpy().astype(np.float32)
        return np.clip(m, 0.0, 1.0).astype(np.float32)

    # 实现 `ViltITMAdapter.predict` 的对象行为，维护该类在模型适配层中的调用契约。
    def predict(self, sample: Sample) -> ModelOutput:
        score = float(self.score_pairs([(sample.image, sample.text)], batch_size=1)[0])
        att = self.attention_map(sample.image, sample.text)
        return ModelOutput(
            text=f"vilt_itm score={score:.4f}",
            score=score,
            attention=att,
            raw={"adapter": "vilt_itm", "device": self._device, "model_name": self._model_name, "score": score},
        )

    # 实现 `ViltITMAdapter.extra_debug` 的对象行为，维护该类在模型适配层中的调用契约。
    def extra_debug(self) -> dict[str, Any]:
        return {"device": str(self._device), "model_name": str(self._model_name)}
