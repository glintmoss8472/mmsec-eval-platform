# 文件说明：该文件属于模型适配层，集中实现 clip hf adapter 相关逻辑。
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from mmsec_eval.model_adapters.image_utils import image_to_rgb01, processor_target_hw, stack_resized_rgb01
from mmsec_eval.plugins.base import ModelAdapter
from mmsec_eval.model_adapters.hf_local import hf_load_failure_message, require_cuda_device, resolve_hf_model_source
from mmsec_eval.types import ModelOutput, Sample

LOG = logging.getLogger(__name__)


# 提取 `feature tensor`，从归档、结果或响应中取出后续流程需要的字段。
def _extract_feature_tensor(out: Any, *, attr_name: str):
    if hasattr(out, "detach") and hasattr(out, "shape"):
        return out
    val = getattr(out, attr_name, None)
    if val is not None:
        return val
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(out, (tuple, list)) and out:
        first = out[0]
        if hasattr(first, "detach") and hasattr(first, "shape"):
            return first
    raise TypeError(f"unsupported CLIP feature output type: {type(out)!r}")


# 实现 `ClipHFAdapter.__init__` 的对象行为，维护该类在模型适配层中的调用契约。
class ClipHFAdapter(ModelAdapter):
    # 封装 ClipHFAdapter.__init__ 的内部步骤，让模型适配层主流程保持清晰并隔离边界细节。
    def __init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._device = require_cuda_device("ClipHFAdapter", torch)
        self._model_name = os.getenv("MMSEC_CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        local_only = os.getenv("MMSEC_HF_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "no"}
        source = resolve_hf_model_source(self._model_name, local_only=local_only, local_dir_name="clip")
        LOG.info("loading CLIP model=%s source=%s device=%s", self._model_name, source, self._device)
        try:
            self._model = CLIPModel.from_pretrained(source, local_files_only=local_only).to(self._device)
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(source, local_files_only=local_only)
        except (OSError, RuntimeError, ValueError) as e:
            raise RuntimeError(
                hf_load_failure_message(
                    adapter_label="CLIP",
                    model_name=self._model_name,
                    source=source,
                    device=self._device,
                    local_only=local_only,
                    cause=e,
                )
            ) from e

    # 实现 `ClipHFAdapter.device` 的对象行为，维护该类在模型适配层中的调用契约。
    @property
    def device(self) -> str:
        return str(self._device)

    # 实现 `ClipHFAdapter._preprocess_images_torch` 的对象行为，维护该类在模型适配层中的调用契约。
    def _preprocess_images_torch(self, images_bchw):
        import torch
        import torch.nn.functional as F

        if images_bchw.ndim != 4:
            raise ValueError("images must be BCHW")
        ip = getattr(self._processor, "image_processor", None)
        size = getattr(ip, "size", None) or {}
        if isinstance(size, dict) and "shortest_edge" in size:
            th = tw = int(size["shortest_edge"])
        elif isinstance(size, dict) and "height" in size and "width" in size:
            th, tw = int(size["height"]), int(size["width"])
        else:
            th = tw = 224
        mean = getattr(ip, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        std = getattr(ip, "image_std", [0.26862954, 0.26130258, 0.27577711])
        x = images_bchw.float().to(self._device).clamp(0.0, 1.0)
        if int(x.shape[2]) != int(th) or int(x.shape[3]) != int(tw):
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        mean_t = torch.tensor(mean, device=self._device, dtype=x.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=self._device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean_t) / std_t

    # 实现 `ClipHFAdapter._target_hw` 的对象行为，维护该类在模型适配层中的调用契约。
    def _target_hw(self) -> tuple[int, int]:
        return processor_target_hw(self._processor, default_hw=(224, 224), prefer_shortest_edge=True)

    # 准备 `numpy 批处理` 数据，补齐后续运行、报告或测试需要的字段。
    def _prepare_numpy_batch(self, images: list[np.ndarray]) -> np.ndarray:
        return stack_resized_rgb01(images, size_hw=self._target_hw())

    # 实现 `ClipHFAdapter._encode_texts_torch` 的对象行为，维护该类在模型适配层中的调用契约。
    def _encode_texts_torch(self, texts: list[str]):
        tok = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(self._device) for k, v in tok.items()}

    # 实现 `ClipHFAdapter.projected_features_torch` 的对象行为，维护该类在模型适配层中的调用契约。
    def projected_features_torch(self, images_bchw, texts: list[str]):
        import torch.nn.functional as F

        pixel_values = self._preprocess_images_torch(images_bchw)
        tok = self._encode_texts_torch(texts)
        img_raw = self._model.get_image_features(pixel_values=pixel_values)
        txt_raw = self._model.get_text_features(**tok)
        img_feat = _extract_feature_tensor(img_raw, attr_name="image_embeds").float()
        txt_feat = _extract_feature_tensor(txt_raw, attr_name="text_embeds").float()
        return F.normalize(img_feat, dim=-1), F.normalize(txt_feat, dim=-1)

    # 计算 `pairs PyTorch`，为指标、风险或调度决策提供数值依据。
    def score_pairs_torch(self, images_bchw, texts: list[str], *, output_attentions: bool = False):
        del output_attentions
        img_feat, txt_feat = self.projected_features_torch(images_bchw, texts)
        return (img_feat * txt_feat).sum(dim=-1)

    # 实现 `ClipHFAdapter.patch_text_similarity_torch` 的对象行为，维护该类在模型适配层中的调用契约。
    def patch_text_similarity_torch(self, images_bchw, texts: list[str]):
        """Per-patch cosine similarity to text embedding (differentiable)."""
        import torch
        import torch.nn.functional as F

        if images_bchw.ndim != 4:
            raise ValueError("images must be BCHW")
        b = int(images_bchw.shape[0])
        if len(texts) == 1 and b > 1:
            texts = [str(texts[0])] * b
        if len(texts) != b:
            raise ValueError(f"text batch mismatch: images={b}, texts={len(texts)}")

        pixel_values = self._preprocess_images_torch(images_bchw)
        tok = self._encode_texts_torch([str(t) for t in texts])

        vision_out = self._model.vision_model(pixel_values=pixel_values, output_attentions=False, return_dict=True)
        tokens = vision_out.last_hidden_state  # [B, 1+N, H]
        patch_tokens = tokens[:, 1:, :]
        patch_proj = self._model.visual_projection(patch_tokens).float()
        patch_proj = F.normalize(patch_proj, dim=-1)

        txt_raw = self._model.get_text_features(**tok)
        txt_feat = _extract_feature_tensor(txt_raw, attr_name="text_embeds").float()
        txt_feat = F.normalize(txt_feat, dim=-1)

        sim = torch.einsum("bnd,bd->bn", patch_proj, txt_feat)
        n = int(sim.shape[1])
        side = int(n**0.5)
        if side * side == n:
            return sim.view(b, side, side)
        return sim

    # 实现 `ClipHFAdapter.patch_text_similarity` 的对象行为，维护该类在模型适配层中的调用契约。
    def patch_text_similarity(self, image: np.ndarray, text: str, eps: float = 1e-8) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        arr = image_to_rgb01(image)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            sim = self.patch_text_similarity_torch(x, [str(text)])
            if sim.ndim == 3:
                m = sim[0].detach().cpu().float()
            else:
                v = sim[0].detach().cpu().float()
                n = int(v.shape[0])
                side = int(n**0.5)
                m = v.view(side, side) if side * side == n else v.view(1, -1)

            h, w = int(arr.shape[0]), int(arr.shape[1])
            m = F.interpolate(m.view(1, 1, m.shape[0], m.shape[1]), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            m = m - m.min()
            m = m / (m.max() + float(eps))
        return m.cpu().numpy().astype(np.float32)

    # 计算 `pairs`，为指标、风险或调度决策提供数值依据。
    def score_pairs(self, pairs: list[tuple[np.ndarray, str]], batch_size: int = 16) -> np.ndarray:
        import torch

        if not pairs:
            return np.zeros((0,), dtype=np.float32)
        out: list[np.ndarray] = []
        step = max(1, int(batch_size))
        for i in range(0, len(pairs), step):
            chunk = pairs[i : i + step]
            images = [np.asarray(im, dtype=np.float32) for im, _ in chunk]
            texts = [str(t) for _, t in chunk]
            x = self._prepare_numpy_batch(images)
            x_t = torch.from_numpy(x).permute(0, 3, 1, 2).to(self._device)
            with torch.no_grad():
                s = self.score_pairs_torch(x_t, texts)
            out.append(s.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0).astype(np.float32)

    # 实现 `ClipHFAdapter.encode_image` 的对象行为，维护该类在模型适配层中的调用契约。
    def encode_image(self, image_np: np.ndarray) -> np.ndarray:
        return self.encode_images_batch([image_np], batch_size=1)[0]

    # 实现 `ClipHFAdapter.encode_text` 的对象行为，维护该类在模型适配层中的调用契约。
    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts_batch([text], batch_size=1)[0]

    # 实现 `ClipHFAdapter.encode_images_batch` 的对象行为，维护该类在模型适配层中的调用契约。
    def encode_images_batch(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        import torch

        if not images:
            return np.zeros((0, 32), dtype=np.float32)
        step = max(1, int(batch_size))
        out_embs: list[np.ndarray] = []
        for i in range(0, len(images), step):
            chunk = images[i : i + step]
            x = self._prepare_numpy_batch(chunk)
            x_t = torch.from_numpy(x).permute(0, 3, 1, 2).to(self._device)
            with torch.no_grad():
                feats, _ = self.projected_features_torch(x_t, [""] * len(chunk))
            out_embs.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out_embs, axis=0).astype(np.float32)

    # 实现 `ClipHFAdapter.encode_texts_batch` 的对象行为，维护该类在模型适配层中的调用契约。
    def encode_texts_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, 32), dtype=np.float32)
        step = max(1, int(batch_size))
        out_embs: list[np.ndarray] = []
        text_probe = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self._device)
        for i in range(0, len(texts), step):
            chunk = [str(t) for t in texts[i : i + step]]
            x_t = text_probe.expand(len(chunk), -1, -1, -1)
            with torch.no_grad():
                _, feats = self.projected_features_torch(x_t, chunk)
            out_embs.append(feats.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out_embs, axis=0).astype(np.float32)

    # 实现 `ClipHFAdapter.attention_map` 的对象行为，维护该类在模型适配层中的调用契约。
    def attention_map(self, image: np.ndarray, text: str, eps: float = 1e-8) -> np.ndarray:
        import torch

        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(self._device)
        x = x.clamp(0.0, 1.0).detach().clone().requires_grad_(True)
        score = self.score_pairs_torch(x, [str(text)]).sum()
        grad = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
        sal = grad.detach().abs().mean(dim=1)[0]
        sal = sal - sal.min()
        sal = sal / (sal.max() + eps)
        return sal.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

    # 实现 `ClipHFAdapter.predict` 的对象行为，维护该类在模型适配层中的调用契约。
    def predict(self, sample: Sample) -> ModelOutput:
        img_emb = self.encode_image(sample.image)
        txt_emb = self.encode_text(sample.text or "")
        sim = float(np.dot(img_emb, txt_emb))
        return ModelOutput(
            text=f"clip similarity={sim:.4f}",
            score=sim,
            embedding=img_emb.astype(np.float32),
            text_embedding=txt_emb.astype(np.float32),
            raw_logits=np.asarray([sim], dtype=np.float32),
            raw={"adapter": "clip_hf", "device": self._device, "model_name": self._model_name, "similarity": sim},
        )

    # 实现 `ClipHFAdapter.extra_debug` 的对象行为，维护该类在模型适配层中的调用契约。
    def extra_debug(self) -> dict[str, Any]:
        return {"device": str(self._device), "model_name": str(self._model_name)}
