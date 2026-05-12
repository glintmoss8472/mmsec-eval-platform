# 文件说明：该文件属于模型适配层，集中实现 blip itm adapter 相关逻辑。
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from mmsec_eval.model_adapters.image_utils import processor_target_hw, stack_resized_rgb01
from mmsec_eval.plugins.base import ModelAdapter
from mmsec_eval.model_adapters.hf_local import hf_load_failure_message, require_cuda_device, resolve_hf_model_source
from mmsec_eval.types import ModelOutput, Sample

LOG = logging.getLogger(__name__)


# 定义 `BlipITMAdapter` 的插件适配边界，把模型、攻击或评测能力暴露为统一接口。
class BlipITMAdapter(ModelAdapter):
    """BLIP image-text retrieval (ITM) adapter, CUDA-only and strict."""

    # 实现 `BlipITMAdapter.__init__` 的对象行为，维护该类在模型适配层中的调用契约。
    def __init__(self) -> None:
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        self._device = require_cuda_device("BlipITMAdapter", torch)
        self._model_name = os.getenv("MMSEC_BLIP_ITM_MODEL_NAME", "Salesforce/blip-itm-base-coco")
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        local_only = os.getenv("MMSEC_HF_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}
        source = resolve_hf_model_source(self._model_name, local_only=local_only, local_dir_name="blip_itm")
        LOG.info("loading BLIP ITM model=%s source=%s device=%s", self._model_name, source, self._device)
        try:
            self._model = BlipForImageTextRetrieval.from_pretrained(
                source, local_files_only=local_only
            ).to(self._device)
            self._model.eval()
            self._processor = BlipProcessor.from_pretrained(source, local_files_only=local_only)
        except (OSError, RuntimeError, ValueError) as e:
            raise RuntimeError(
                hf_load_failure_message(
                    adapter_label="BLIP ITM",
                    model_name=self._model_name,
                    source=source,
                    device=self._device,
                    local_only=local_only,
                    cause=e,
                )
            ) from e

    # 实现 `BlipITMAdapter.device` 的对象行为，维护该类在模型适配层中的调用契约。
    @property
    def device(self) -> str:
        return str(self._device)

    # 实现 `BlipITMAdapter._image_preprocess_torch` 的对象行为，维护该类在模型适配层中的调用契约。
    def _image_preprocess_torch(self, images_bchw):
        import torch
        import torch.nn.functional as F

        if images_bchw.ndim != 4:
            raise ValueError("images must be BCHW")
        ip = getattr(self._processor, "image_processor", None)
        size = getattr(ip, "size", None) or {}
        if isinstance(size, dict) and "height" in size and "width" in size:
            th, tw = int(size["height"]), int(size["width"])
        elif isinstance(size, dict) and "shortest_edge" in size:
            th = tw = int(size["shortest_edge"])
        else:
            th = tw = 384
        x = images_bchw.float().to(self._device).clamp(0.0, 1.0)
        x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        mean = getattr(ip, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(ip, "image_std", [0.5, 0.5, 0.5])
        mean_t = torch.tensor(mean, device=self._device, dtype=x.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=self._device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean_t) / std_t

    # 实现 `BlipITMAdapter._target_hw` 的对象行为，维护该类在模型适配层中的调用契约。
    def _target_hw(self) -> tuple[int, int]:
        return processor_target_hw(self._processor, default_hw=(384, 384))

    # 准备 `numpy 批处理` 数据，补齐后续运行、报告或测试需要的字段。
    def _prepare_numpy_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return stack_resized_rgb01(images, size_hw=self._target_hw())

    # 实现 `BlipITMAdapter._encode_texts_torch` 的对象行为，维护该类在模型适配层中的调用契约。
    def _encode_texts_torch(self, texts: list[str]):
        enc = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        out: dict[str, Any] = {}
        for k, v in enc.items():
            out[k] = v.to(self._device)
        return out

    # 计算 `pairs PyTorch`，为指标、风险或调度决策提供数值依据。
    def score_pairs_torch(self, images_bchw, texts: list[str], *, output_attentions: bool = False):
        import torch

        pixel_values = self._image_preprocess_torch(images_bchw)
        enc = self._encode_texts_torch(texts)
        out = self._model(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            pixel_values=pixel_values,
            use_itm_head=True,
            output_attentions=bool(output_attentions),
            return_dict=True,
        )
        logits = out.itm_score.float()
        return torch.softmax(logits, dim=-1)[:, 1]

    # 实现 `BlipITMAdapter.projected_features_torch` 的对象行为，维护该类在模型适配层中的调用契约。
    def projected_features_torch(self, images_bchw, texts: list[str]):
        import torch.nn.functional as F

        pixel_values = self._image_preprocess_torch(images_bchw)
        enc = self._encode_texts_torch(texts)
        vision_out = self._model.vision_model(pixel_values=pixel_values, output_attentions=False, return_dict=True)
        image_embeds = vision_out.last_hidden_state
        img_feat = F.normalize(self._model.vision_proj(image_embeds[:, 0, :]).float(), dim=-1)
        text_out = self._model.text_encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            return_dict=True,
        )
        q = text_out.last_hidden_state
        txt_feat = F.normalize(self._model.text_proj(q[:, 0, :]).float(), dim=-1)
        return img_feat, txt_feat

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

    # 实现 `BlipITMAdapter.fused_embedding` 的对象行为，维护该类在模型适配层中的调用契约。
    def fused_embedding(self, image: np.ndarray, text: str) -> np.ndarray:
        import torch

        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            img_feat, txt_feat = self.projected_features_torch(x, [str(text)])
            fused = torch.cat([img_feat, txt_feat], dim=-1)
        return fused[0].detach().cpu().numpy().astype(np.float32)

    # 实现 `BlipITMAdapter.attention_map` 的对象行为，维护该类在模型适配层中的调用契约。
    def attention_map(self, image: np.ndarray, text: str, eps: float = 1e-8) -> np.ndarray:
        import math
        import torch

        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(self._device)
        pixel_values = self._image_preprocess_torch(x)
        enc = self._encode_texts_torch([str(text)])
        with torch.no_grad():
            out = self._model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                pixel_values=pixel_values,
                use_itm_head=True,
                output_attentions=True,
                return_dict=True,
            )
            attns = out.attentions or []
            if not attns:
                raise RuntimeError("BLIP attentions are unavailable")
            last = attns[-1].float()
            cls_to = last.mean(dim=1)[:, 0, 1:]
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

    # 实现 `BlipITMAdapter.predict` 的对象行为，维护该类在模型适配层中的调用契约。
    def predict(self, sample: Sample) -> ModelOutput:
        score = float(self.score_pairs([(sample.image, sample.text)], batch_size=1)[0])
        att = self.attention_map(sample.image, sample.text)
        return ModelOutput(
            text=f"blip_itm score={score:.4f}",
            score=score,
            attention=att,
            raw={"adapter": "blip_itm", "device": self._device, "model_name": self._model_name, "score": score},
        )

    # 实现 `BlipITMAdapter.extra_debug` 的对象行为，维护该类在模型适配层中的调用契约。
    def extra_debug(self) -> dict[str, Any]:
        return {"device": str(self._device), "model_name": str(self._model_name)}
